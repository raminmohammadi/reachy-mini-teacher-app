"""Local pipeline handler: Faster-Whisper STT → Ollama LLM → Kokoro-ONNX TTS.

Pipeline flow
─────────────
1. Mic audio is buffered frame-by-frame.
2. Energy-based VAD detects speech boundaries (silence after speech).
3. The accumulated speech buffer is transcribed with Faster-Whisper.
4. The transcript is sent to Ollama (streaming chat, tool calling supported).
5. Each assistant text chunk is synthesised by Kokoro-ONNX TTS.
6. Resulting PCM audio is enqueued for playback via emit().
"""

from __future__ import annotations

import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample
from fastrtc import AdditionalOutputs, AsyncStreamHandler, audio_to_int16, wait_for_item

from reachy_mini_teacher_app.config import config
from reachy_mini_teacher_app.prompts import get_session_instructions
from reachy_mini_teacher_app.tools.core_tools import ToolDependencies, get_tool_specs, dispatch_tool_call


logger = logging.getLogger(__name__)

# Audio constants
MIC_SAMPLE_RATE = 16000   # Whisper expects 16 kHz mono float32
TTS_SAMPLE_RATE = 24000   # Kokoro default output rate

# VAD tuning
SILENCE_THRESHOLD_RMS = 150    # RMS below this → silence
SPEECH_START_FRAMES = 5        # consecutive loud frames to begin capture
SILENCE_END_FRAMES = 40        # consecutive quiet frames to end utterance (≈0.4 s at 100 Hz)
MIN_SPEECH_FRAMES = 10         # ignore very short bursts


class LocalPipelineHandler(AsyncStreamHandler):
    """Turn-based STT→LLM→TTS pipeline, no cloud audio API required."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=TTS_SAMPLE_RATE,
            input_sample_rate=MIC_SAMPLE_RATE,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()

        # VAD state
        self._speech_frames: List[NDArray] = []
        self._loud_count = 0
        self._quiet_count = 0
        self._recording = False

        # STT/LLM/TTS handles (lazy-loaded on start_up)
        self._whisper: Any = None
        self._kokoro: Any = None
        self._conversation: List[Dict[str, Any]] = []
        self._shutdown_requested = False
        self._clear_queue: Optional[Any] = None

    def copy(self) -> "LocalPipelineHandler":
        return LocalPipelineHandler(self.deps, self.gradio_mode)

    # ------------------------------------------------------------------
    # FastRTC interface
    # ------------------------------------------------------------------

    async def receive(self, frame: Tuple[int, NDArray]) -> None:
        """Buffer mic frame and run VAD; trigger pipeline when utterance ends."""
        input_sr, audio_frame = frame

        # Flatten to mono 1-D
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample to 16 kHz
        if input_sr != MIC_SAMPLE_RATE:
            audio_frame = resample(
                audio_frame,
                int(len(audio_frame) * MIC_SAMPLE_RATE / input_sr),
            ).astype(np.float32)

        rms = float(np.sqrt(np.mean(audio_frame.astype(np.float64) ** 2)))

        if rms > SILENCE_THRESHOLD_RMS:
            self._loud_count += 1
            self._quiet_count = 0
            if self._loud_count >= SPEECH_START_FRAMES:
                self._recording = True
        else:
            self._quiet_count += 1
            if self._recording:
                self._loud_count = 0

        if self._recording:
            self._speech_frames.append(audio_frame)

        if self._recording and self._quiet_count >= SILENCE_END_FRAMES:
            speech_buffer = self._speech_frames[:]
            self._speech_frames = []
            self._recording = False
            self._loud_count = 0
            self._quiet_count = 0

            if len(speech_buffer) >= MIN_SPEECH_FRAMES:
                asyncio.create_task(self._run_pipeline(speech_buffer))

    async def emit(self) -> Any:
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        self._shutdown_requested = True

    # ------------------------------------------------------------------
    # Lazy model initialisation
    # ------------------------------------------------------------------

    async def start_up(self) -> None:
        """Load STT and TTS models, then wait forever (pipeline is event-driven)."""
        await asyncio.to_thread(self._load_models)
        logger.info("Local pipeline ready (whisper=%s, kokoro voice=%s)",
                    config.WHISPER_MODEL, config.KOKORO_VOICE)
        # Keep the coroutine alive so FastRTC doesn't tear down the handler
        while not self._shutdown_requested:
            await asyncio.sleep(0.5)

    def _load_models(self) -> None:
        """Load Faster-Whisper and Kokoro-ONNX (blocking; called in thread)."""
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        self._whisper = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        from kokoro_onnx import Kokoro  # type: ignore[import-untyped]
        self._kokoro = Kokoro(config.KOKORO_VOICE, config.KOKORO_LANG)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _run_pipeline(self, speech_frames: List[NDArray]) -> None:
        """STT → LLM → TTS for one complete utterance."""
        if self._whisper is None or self._kokoro is None:
            logger.warning("Models not loaded yet; dropping utterance")
            return

        # --- 1. STT ---
        pcm = np.concatenate(speech_frames).astype(np.float32)
        # Normalise to [-1, 1] if it's int16 range
        if pcm.max() > 1.0:
            pcm = pcm / 32768.0

        segments, _ = await asyncio.to_thread(
            lambda: self._whisper.transcribe(pcm, language="en", vad_filter=True)
        )
        transcript = " ".join(s.text.strip() for s in segments).strip()
        if not transcript:
            return

        logger.info("STT: %r", transcript)
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

        # --- 2. Build LLM messages ---
        if not self._conversation:
            system_prompt = get_session_instructions()
            self._conversation.append({"role": "system", "content": system_prompt})
        self._conversation.append({"role": "user", "content": transcript})

        tool_specs = get_tool_specs()
        ollama_tools = self._build_ollama_tools(tool_specs)

        # --- 3. LLM call (Ollama, tool loop) ---
        assistant_text = await self._ollama_turn(ollama_tools)

        if assistant_text:
            self._conversation.append({"role": "assistant", "content": assistant_text})
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": assistant_text})
            )

        # --- 4. TTS ---
        if assistant_text:
            await self._synthesise_and_enqueue(assistant_text)

    # ------------------------------------------------------------------
    # Ollama integration
    # ------------------------------------------------------------------

    def _build_ollama_tools(self, tool_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "parameters": s.get("parameters", {}),
                },
            }
            for s in tool_specs
        ]

    async def _ollama_turn(self, ollama_tools: List[Dict[str, Any]]) -> str:
        """Run one Ollama chat turn (handles tool calls recursively)."""
        import ollama as ollama_lib  # type: ignore[import-untyped]

        client = ollama_lib.AsyncClient(host=config.OLLAMA_BASE_URL)
        MAX_TOOL_ROUNDS = 5
        full_reply = ""

        for _ in range(MAX_TOOL_ROUNDS):
            response = await client.chat(
                model=config.OLLAMA_MODEL,
                messages=self._conversation,
                tools=ollama_tools or None,
            )
            msg = response.message
            tool_calls = getattr(msg, "tool_calls", None)

            if tool_calls:
                for tc in tool_calls:
                    fn = tc.function
                    tool_name = fn.name
                    args_dict = dict(fn.arguments) if fn.arguments else {}
                    logger.info("Tool call: %s %s", tool_name, args_dict)
                    result = await dispatch_tool_call(tool_name, json.dumps(args_dict), self.deps)
                    logger.info("Tool result: %s → %s", tool_name, result)

                    self._conversation.append({"role": "tool", "name": tool_name, "content": json.dumps(result)})
            else:
                content = (msg.content or "").strip()
                full_reply = content
                break

        return full_reply

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    async def _synthesise_and_enqueue(self, text: str) -> None:
        """Convert text to PCM via Kokoro-ONNX and enqueue for playback."""
        try:
            samples, sample_rate = await asyncio.to_thread(
                lambda: self._kokoro.create(text, voice=config.KOKORO_VOICE, speed=1.0, lang=config.KOKORO_LANG)
            )
            # Kokoro returns float32 PCM; convert to int16
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).reshape(1, -1)
            hw = self.deps.head_wobbler
            if hw is not None:
                import base64
                hw.feed(base64.b64encode(pcm.tobytes()).decode())
            await self.output_queue.put((sample_rate, pcm))
        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)

