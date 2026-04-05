"""OpenAI Realtime API handler for real-time bidirectional audio + tools."""

from __future__ import annotations

import json
import base64
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample
from fastrtc import AdditionalOutputs, AsyncStreamHandler, audio_to_int16, wait_for_item

from reachy_mini_teacher_app.config import config
from reachy_mini_teacher_app.prompts import get_session_instructions, get_session_voice
from reachy_mini_teacher_app.session_db import SessionDB
from reachy_mini_teacher_app.tools.core_tools import ToolDependencies, get_tool_specs, reinitialize_tools, dispatch_tool_call
from reachy_mini_teacher_app.tools.background_tool_manager import BackgroundToolManager, ToolCallRoutine, ToolNotification


logger = logging.getLogger(__name__)

OPENAI_SAMPLE_RATE = 24000  # OpenAI Realtime uses 24 kHz PCM16 for both input and output


# ---------------------------------------------------------------------------
# Tool spec conversion: internal format → OpenAI Realtime function tools
# ---------------------------------------------------------------------------

def _tool_specs_to_openai(tool_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert internal tool specs to OpenAI Realtime function tool format."""
    tools = []
    for spec in tool_specs:
        tools.append({
            "type": "function",
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", {}),
        })
    return tools


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class OpenAIRealtimeHandler(AsyncStreamHandler):
    """AsyncStreamHandler that streams real-time audio through OpenAI Realtime API."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPENAI_SAMPLE_RATE,
            input_sample_rate=OPENAI_SAMPLE_RATE,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._audio_in_queue: asyncio.Queue[Tuple[int, NDArray]] = asyncio.Queue()
        self._connection: Any = None
        self._shutdown_requested = False
        self._clear_queue: Optional[Any] = None
        self.tool_manager = BackgroundToolManager()
        # Session database for conversation history
        self._db = SessionDB(config.SESSION_DB_PATH)
        self._session_id: Optional[int] = None

    def copy(self) -> "OpenAIRealtimeHandler":
        return OpenAIRealtimeHandler(self.deps, self.gradio_mode)

    # ------------------------------------------------------------------
    # FastRTC interface
    # ------------------------------------------------------------------

    async def receive(self, frame: Tuple[int, NDArray]) -> None:
        await self._audio_in_queue.put(frame)

    async def emit(self) -> Any:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        self._shutdown_requested = True
        await self.tool_manager.shutdown()
        # Generate summary and end session in database
        if self._session_id is not None:
            summary = await self._generate_session_summary()
            self._db.end_session(self._session_id, summary=summary)
            logger.info("Database session %d ended (summary=%s)", self._session_id, "yes" if summary else "no")
        self._db.close()
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:
                pass

    async def _generate_session_summary(self) -> Optional[str]:
        """Generate a summary of the current session for the database."""
        if self._session_id is None:
            return None
        try:
            from reachy_mini_teacher_app.session_summarizer import generate_session_summary
            messages = self._db.get_session_messages(self._session_id)
            return await generate_session_summary(messages)
        except Exception as e:
            logger.error("Failed to generate session summary: %s", e)
            return None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _build_openai_session_payload(self) -> Dict[str, Any]:
        """Build the session.update payload from current profile and config."""
        # Build recap from all known users for prompt injection
        recap_parts: List[str] = []
        for user in self._db.list_users():
            user_recap = self._db.build_recap_for_user(user["id"])
            if user_recap:
                recap_parts.append(f"کاربر «{user['name']}»:\n{user_recap}")
        recap = "\n\n".join(recap_parts) if recap_parts else None

        instructions = get_session_instructions(recap=recap)
        voice = get_session_voice(default="cedar")

        # Inject current session transcript so the AI doesn't lose context on reconnect
        if self._session_id is not None:
            transcript = self._db.build_current_session_transcript(self._session_id)
            if transcript:
                instructions = (
                    "## CRITICAL: THIS IS A RESUMED SESSION\n"
                    "The connection was temporarily interrupted. You MUST:\n"
                    "- Do NOT greet, introduce yourself, or say hello again\n"
                    "- Do NOT ask the user's name again\n"
                    "- Do NOT start a new conversation\n"
                    "- Simply continue naturally from where you left off\n"
                    "- If you were waiting for the user to respond, wait silently\n\n"
                    + instructions
                    + "\n\n## مکالمه جاری (برای ادامه):\n"
                    + transcript
                )
                logger.info("Injected resumed-session directive + %d-line transcript", transcript.count('\n') + 1)

        tool_specs = get_tool_specs()
        openai_tools = _tool_specs_to_openai(tool_specs) if tool_specs else []

        # Build VAD turn detection config
        turn_detection: Dict[str, Any] = {"type": "server_vad"}
        silence_ms = config.VAD_SILENCE_DURATION_MS
        prefix_ms = config.VAD_PREFIX_PADDING_MS
        if silence_ms > 0:
            turn_detection["silence_duration_ms"] = silence_ms
        if prefix_ms > 0:
            turn_detection["prefix_padding_ms"] = prefix_ms
        if silence_ms > 0 or prefix_ms > 0:
            turn_detection["threshold"] = 0.3
            logger.info("VAD tuning: silence=%dms prefix=%dms threshold=0.3", silence_ms, prefix_ms)

        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": instructions,
                "voice": voice,
                "tools": openai_tools,
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000,
                        },
                        "transcription": {
                            "model": "gpt-4o-mini-transcribe",
                        },
                        "turn_detection": turn_detection,
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000,
                        },
                    },
                },
                "output_modalities": ["audio", "text"],
            },
        }

    async def start_up(self) -> None:
        """Connect to OpenAI Realtime and run receive/send loops until shutdown."""
        from openai import AsyncOpenAI

        api_key = config.OPENAI_API_KEY
        if not api_key:
            logger.error("OPENAI_API_KEY not set – cannot start OpenAI Realtime handler.")
            return

        client = AsyncOpenAI(api_key=api_key)

        # Set up profile switch event so tools can trigger a reconnect
        self._profile_switch_event = asyncio.Event()
        self.deps.profile_switch_event = self._profile_switch_event

        # Start a new database session
        self._session_id = self._db.start_session()
        logger.info("Database session started: %d", self._session_id)

        head_wobbler = self.deps.head_wobbler
        if head_wobbler is not None:
            head_wobbler.start()

        # Start tool manager with callback
        self.tool_manager.start_up(tool_callbacks=[self._on_tool_complete])

        reconnect_delay = 2.0
        while not self._shutdown_requested:
            # (Re)build config from current profile on every reconnect
            if self._profile_switch_event.is_set():
                self._profile_switch_event.clear()
                reinitialize_tools()
                # Start a fresh DB session so old transcript doesn't bleed into new persona
                if self._session_id is not None:
                    self._db.end_session(self._session_id, summary="Profile switched")
                self._session_id = self._db.start_session()
                logger.info("Profile switch detected — new DB session %d", self._session_id)

            session_payload = self._build_openai_session_payload()

            try:
                async with client.realtime.connect(model=config.OPENAI_MODEL) as conn:
                    self._connection = conn

                    # Configure the session
                    await conn.send(session_payload)
                    logger.info("OpenAI Realtime connected (model=%s)", config.OPENAI_MODEL)

                    send_task = asyncio.create_task(self._send_loop(conn))
                    recv_task = asyncio.create_task(self._recv_loop(conn))
                    try:
                        done, pending = await asyncio.wait(
                            [send_task, recv_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for t in pending:
                            t.cancel()
                            try:
                                await t
                            except (asyncio.CancelledError, Exception):
                                pass
                    except asyncio.CancelledError:
                        send_task.cancel()
                        recv_task.cancel()
                        break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("OpenAI Realtime session error: %s", e)

            if self._shutdown_requested:
                break

            # Drain stale mic frames
            while not self._audio_in_queue.empty():
                try:
                    self._audio_in_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            logger.info("Reconnecting to OpenAI Realtime in %.1fs …", reconnect_delay)
            await asyncio.sleep(reconnect_delay)

        self._connection = None
        if head_wobbler is not None:
            head_wobbler.stop()

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    async def _send_loop(self, conn: Any) -> None:
        """Read buffered mic frames and forward them to OpenAI as base64 PCM16."""
        while not self._shutdown_requested:
            # Break out if a profile switch was requested
            if self._profile_switch_event.is_set():
                logger.info("Profile switch detected in _send_loop — breaking to reconnect")
                return
            try:
                input_sr, audio_frame = await asyncio.wait_for(
                    self._audio_in_queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue

            if input_sr == 0 or len(audio_frame) == 0:
                continue

            # Flatten stereo → mono
            if audio_frame.ndim == 2:
                if audio_frame.shape[1] > audio_frame.shape[0]:
                    audio_frame = audio_frame.T
                if audio_frame.shape[1] > 1:
                    audio_frame = audio_frame[:, 0]

            # Resample to 24 kHz if needed
            if input_sr != OPENAI_SAMPLE_RATE:
                num_samples = int(len(audio_frame) * OPENAI_SAMPLE_RATE / input_sr)
                if num_samples > 0:
                    audio_frame = resample(audio_frame, num_samples)
                else:
                    continue

            pcm_int16 = audio_to_int16(audio_frame)
            b64_audio = base64.b64encode(pcm_int16.tobytes()).decode("ascii")
            try:
                await conn.send({
                    "type": "input_audio_buffer.append",
                    "audio": b64_audio,
                })
            except Exception as e:
                logger.debug("input_audio_buffer.append error: %s", e)

    async def _recv_loop(self, conn: Any) -> None:
        """Receive OpenAI Realtime events: audio, transcripts, tool calls."""
        try:
            async for event in conn:
                if self._shutdown_requested:
                    break

                event_type = event.type

                # Audio delta
                if event_type == "response.output_audio.delta":
                    audio_bytes = base64.b64decode(event.delta)
                    audio_arr = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
                    # Feed head wobbler
                    hw = self.deps.head_wobbler
                    if hw is not None:
                        hw.feed(event.delta)
                    await self.output_queue.put((OPENAI_SAMPLE_RATE, audio_arr))

                # Audio done
                elif event_type == "response.output_audio.done":
                    hw = self.deps.head_wobbler
                    if hw is not None:
                        hw.reset()

                # Text transcript (assistant)
                elif event_type == "response.output_audio_transcript.done":
                    text = getattr(event, "transcript", "")
                    if text:
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": text})
                        )
                        # Log assistant text to DB
                        if self._session_id is not None:
                            self._db.add_message(self._session_id, "assistant", text)

                # User speech transcript
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    text = getattr(event, "transcript", "")
                    if text:
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "user", "content": text})
                        )
                        # Log user speech to DB
                        if self._session_id is not None:
                            self._db.add_message(self._session_id, "user", text)

                # Turn complete — mark activity
                elif event_type == "response.done":
                    mm = self.deps.movement_manager
                    if mm is not None:
                        mm.mark_activity()

                # Function call arguments done — dispatch tool
                elif event_type == "response.function_call_arguments.done":
                    tool_name = event.name
                    call_id = event.call_id
                    args_json = event.arguments

                    logger.info("Tool call: %s call_id=%s args=%s", tool_name, call_id, args_json[:200])

                    bg_tool = await self.tool_manager.start_tool(
                        call_id=call_id,
                        tool_call_routine=ToolCallRoutine(
                            tool_name=tool_name,
                            args_json_str=args_json,
                            deps=self.deps,
                        ),
                        is_idle_tool_call=False,
                    )

                    await self.output_queue.put(
                        AdditionalOutputs(
                            {"role": "assistant", "content": f"🛠️ Used tool {tool_name}. ID: {bg_tool.tool_id}"}
                        )
                    )

                # Error
                elif event_type == "error":
                    error_info = getattr(event, "error", event)
                    logger.error("OpenAI Realtime error: %s", error_info)

        except asyncio.CancelledError:
            logger.debug("_recv_loop cancelled")
        except Exception as e:
            logger.error("_recv_loop error: %s", e)
        else:
            logger.warning("_recv_loop: event iterator ended (session closed by server)")

    async def _on_tool_complete(self, notification: ToolNotification) -> None:
        """Send function call output back to OpenAI and trigger a new response."""
        conn = self._connection
        if conn is None:
            logger.warning("Tool %s completed but no active OpenAI connection", notification.tool_name)
            return

        # Build output payload — sanitize large values
        if notification.error:
            output_str = json.dumps({"error": notification.error})
        elif notification.result:
            sanitized = {
                k: v for k, v in notification.result.items()
                if not (isinstance(v, str) and len(v) > 10_000)
            }
            output_str = json.dumps(sanitized or {"result": "ok"})
        else:
            output_str = json.dumps({"result": "ok"})

        try:
            # Create the function call output conversation item
            await conn.send({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": notification.id,
                    "output": output_str,
                },
            })
            # Ask the model to generate a response based on the tool output
            await conn.send({"type": "response.create"})
            logger.info("Sent tool response for %s (call_id=%s)", notification.tool_name, notification.id)
        except Exception as e:
            logger.error("Failed to send tool response for %s: %s", notification.tool_name, e)

