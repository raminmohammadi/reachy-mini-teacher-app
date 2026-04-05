"""Gemini Multimodal Live API handler for real-time bidirectional audio + vision + tools."""

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

GEMINI_INPUT_SAMPLE_RATE = 16000   # Hz – Gemini Live expects 16 kHz PCM input
GEMINI_OUTPUT_SAMPLE_RATE = 24000  # Hz – Gemini Live emits 24 kHz PCM output


# ---------------------------------------------------------------------------
# Tool spec conversion: OpenAI format → Gemini FunctionDeclaration list
# ---------------------------------------------------------------------------

def _type_str_to_gemini(type_str: str, gtypes: Any) -> Any:
    mapping = {
        "string": gtypes.Type.STRING,
        "number": gtypes.Type.NUMBER,
        "integer": gtypes.Type.INTEGER,
        "boolean": gtypes.Type.BOOLEAN,
        "array": gtypes.Type.ARRAY,
        "object": gtypes.Type.OBJECT,
    }
    return mapping.get((type_str or "string").lower(), gtypes.Type.STRING)


def _tool_specs_to_gemini(tool_specs: List[Dict[str, Any]], gtypes: Any) -> List[Any]:
    """Convert tool spec list (OpenAI-style JSON Schema) to Gemini FunctionDeclaration objects."""
    declarations = []
    for spec in tool_specs:
        params_schema = spec.get("parameters", {})
        props = params_schema.get("properties", {})
        required = params_schema.get("required", [])

        gemini_props = {
            k: gtypes.Schema(
                type=_type_str_to_gemini(v.get("type", "string"), gtypes),
                description=v.get("description", ""),
                enum=v.get("enum") or None,
            )
            for k, v in props.items()
        }

        declarations.append(
            gtypes.FunctionDeclaration(
                name=spec["name"],
                description=spec.get("description", ""),
                parameters=gtypes.Schema(
                    type=gtypes.Type.OBJECT,
                    properties=gemini_props,
                    required=required,
                ) if gemini_props else None,
            )
        )
    return declarations


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class GeminiLiveHandler(AsyncStreamHandler):
    """AsyncStreamHandler that streams real-time audio through Gemini Live API."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=GEMINI_OUTPUT_SAMPLE_RATE,
            input_sample_rate=GEMINI_INPUT_SAMPLE_RATE,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._audio_in_queue: asyncio.Queue[Tuple[int, NDArray]] = asyncio.Queue()
        self._session: Any = None
        self._shutdown_requested = False
        self._clear_queue: Optional[Any] = None
        self.tool_manager = BackgroundToolManager()
        # Session database for conversation history
        self._db = SessionDB(config.SESSION_DB_PATH)
        self._session_id: Optional[int] = None
        # Shared state dict exposed to tools so they can read/write session info
        self._session_state: Dict[str, Any] = {"session_id": None}
        deps.session_db = self._db
        deps.session_state = self._session_state
        # Session resumption token — keeps Gemini context across reconnects
        self._resumption_token: Optional[str] = None
        # Track whether this is the first connection (for intro vs resume behavior)
        self._is_first_connection = True
        # Cached config — reused across reconnects so Gemini sees identical config
        self._cached_live_config: Optional[Any] = None
        self._needs_config_rebuild = True
        # Set to True after a deliberate profile switch so the kickoff turn is sent
        # and the reconnect delay is shortened.
        self._profile_just_switched = False
        # Daily curriculum plan — set once when the session starts
        self._daily_plan: Optional[Dict[str, Any]] = None
        self._today_date: Optional[str] = None
        # Known user name — looked up at session start, shown in prompt
        self._known_user_name: Optional[str] = None

    def copy(self) -> "GeminiLiveHandler":
        return GeminiLiveHandler(self.deps, self.gradio_mode)

    # ------------------------------------------------------------------
    # FastRTC interface
    # ------------------------------------------------------------------

    async def receive(self, frame: Tuple[int, NDArray]) -> None:
        """Buffer incoming microphone audio; sent to Gemini by _send_loop."""
        await self._audio_in_queue.put(frame)

    async def emit(self) -> Any:
        """Return the next audio frame or transcript for the caller."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._shutdown_requested = True
        await self.tool_manager.shutdown()
        # Generate summary, evaluate pass/fail, end session
        if self._session_id is not None:
            result = await self._generate_session_summary()
            summary = result.get("summary") if result else None
            passed = result.get("passed", False) if result else False
            self._db.end_session(self._session_id, summary=summary)
            logger.info("Database session %d ended (summary=%s)", self._session_id, "yes" if summary else "no")
            # Record whether the student passed today's unit
            if self._daily_plan is not None and self._today_date is not None:
                self._db.mark_daily_plan_result(self._today_date, passed)
        self._db.close()
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass

    async def _generate_session_summary(self) -> Optional[Dict[str, Any]]:
        """Generate a summary and pass/fail verdict for the current session."""
        if self._session_id is None:
            return None
        try:
            from reachy_mini_teacher_app.session_summarizer import generate_session_summary
            messages = self._db.get_session_messages(self._session_id)
            return await generate_session_summary(messages, daily_plan=self._daily_plan)
        except Exception as e:
            logger.error("Failed to generate session summary: %s", e)
            return None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _get_resumption_config(self, gtypes: Any) -> Any:
        """Create SessionResumptionConfig dynamically to support different SDK versions."""
        kwargs = {}
        try:
            fields = gtypes.SessionResumptionConfig.model_fields
        except AttributeError:
            try:
                fields = gtypes.SessionResumptionConfig.__fields__
            except AttributeError:
                fields = {"resumption_token": None, "handle": None}
        
        if "resumption_token" in fields:
            kwargs["resumption_token"] = self._resumption_token
        elif "handle" in fields:
            kwargs["handle"] = self._resumption_token
        else:
            kwargs["resumption_token"] = self._resumption_token
            
        return gtypes.SessionResumptionConfig(**kwargs)

    def _build_live_config(self, gtypes: Any) -> Any:
        """Build or return cached LiveConnectConfig.

        The config is only rebuilt when ``_needs_config_rebuild`` is True
        (first connect or after a profile switch), OR when we don't have a 
        resumption token (meaning we are starting fresh and can inject the transcript).
        On normal reconnects with a token, the cached config is reused so that 
        Gemini sees an **identical** config and session resumption works reliably.
        """
        rebuild = self._needs_config_rebuild or self._cached_live_config is None or not self._resumption_token

        if rebuild:
            # Fetch real summaries from previous sessions (works without user-ID linkage).
            recap = self._db.get_recent_session_recap(current_session_id=self._session_id)

            # Inject today's daily plan if one is active (english_teacher profile)
            daily_plan_text: Optional[str] = None
            if self._daily_plan is not None and self._today_date is not None:
                daily_plan_text = self._db.get_daily_plan_for_prompt(self._today_date)

            instructions = get_session_instructions(
                recap=recap,
                daily_plan=daily_plan_text,
                user_name=self._known_user_name,
            )

            # On a mid-session reconnect (no resumption token, but session already has messages)
            # inject the transcript so the AI continues naturally instead of restarting.
            if self._session_id is not None and not self._profile_just_switched:
                transcript = self._db.build_current_session_transcript(self._session_id)
                if transcript:
                    instructions = (
                        "## این اتصال ادامه یک مکالمه قطع‌شده است.\n"
                        "- سلام نکن، خودت را معرفی نکن، و اسم کاربر را دوباره نپرس.\n"
                        "- مکالمه را از همان‌جا که ماند ادامه بده.\n\n"
                        + instructions
                        + "\n\n## مکالمه جاری (برای ادامه):\n"
                        + transcript
                    )
                    logger.info("Injected resumed-session transcript (%d lines)", transcript.count('\n') + 1)

            # Minimal, non-conflicting behavioral guardrails.
            instructions += (
                "\n\n---\n"
                "قوانین رفتاری:\n"
                "- پاسخ‌ها را کوتاه نگه دار (حداکثر ۲ جمله).\n"
                "- سوالی که در همین جلسه پرسیده‌ای دوباره نپرس.\n"
                "- اسم کاربر را در هر جمله تکرار نکن.\n"
                "- اگر خلاصه جلسات قبلی داری، از آن استفاده کن و مستقیماً ادامه بده.\n"
                "---\n"
            )

            voice_name = get_session_voice(default="Aoede")

            tool_specs = get_tool_specs()
            gemini_tools = _tool_specs_to_gemini(tool_specs, gtypes) if tool_specs else []

            # Build realtime input config for VAD tuning
            realtime_input_kwargs: Dict[str, Any] = {}
            silence_ms = config.VAD_SILENCE_DURATION_MS
            prefix_ms = config.VAD_PREFIX_PADDING_MS
            if silence_ms > 0 or prefix_ms > 0:
                aad_kwargs: Dict[str, Any] = {}
                if silence_ms > 0:
                    aad_kwargs["silenceDurationMs"] = silence_ms
                if prefix_ms > 0:
                    aad_kwargs["prefixPaddingMs"] = prefix_ms
                aad_kwargs["endOfSpeechSensitivity"] = "END_SENSITIVITY_LOW"
                realtime_input_kwargs["automaticActivityDetection"] = gtypes.AutomaticActivityDetection(**aad_kwargs)
                logger.info("VAD tuning: silence=%dms prefix=%dms sensitivity=LOW", silence_ms, prefix_ms)

            self._cached_live_config = gtypes.LiveConnectConfig(
                response_modalities=[gtypes.Modality.AUDIO],
                system_instruction=gtypes.Content(parts=[gtypes.Part(text=instructions)]),
                tools=[gtypes.Tool(function_declarations=gemini_tools)] if gemini_tools else [],
                speech_config=gtypes.SpeechConfig(
                    voice_config=gtypes.VoiceConfig(
                        prebuilt_voice_config=gtypes.PrebuiltVoiceConfig(voice_name=voice_name),
                    ),
                ),
                realtimeInputConfig=gtypes.RealtimeInputConfig(**realtime_input_kwargs) if realtime_input_kwargs else None,
                input_audio_transcription=gtypes.AudioTranscriptionConfig(),
                output_audio_transcription=gtypes.AudioTranscriptionConfig(),
                session_resumption=self._get_resumption_config(gtypes),
                # context_window_compression intentionally omitted:
                # SlidingWindow() with default token threshold closes the session
                # every ~10-15 s, causing the AI to resume mid-question and repeat
                # the same question endlessly.
            )
            self._needs_config_rebuild = False
            logger.info("Built new LiveConnectConfig (resumption_token=%s)", "yes" if self._resumption_token else "no")
        else:
            # Only update the resumption handle — everything else stays identical
            self._cached_live_config.session_resumption = self._get_resumption_config(gtypes)

        return self._cached_live_config

    async def start_up(self) -> None:
        """Connect to Gemini Live and run receive/send loops until shutdown."""
        from google import genai
        from google.genai import types as gtypes

        api_key = config.GEMINI_API_KEY
        if not api_key:
            logger.error("GEMINI_API_KEY not set – cannot start Gemini Live handler.")
            return

        client = genai.Client(api_key=api_key)

        # Set up profile switch event so tools can trigger a reconnect
        self._profile_switch_event = asyncio.Event()
        self.deps.profile_switch_event = self._profile_switch_event

        # Start a new database session
        self._session_id = self._db.start_session()
        self._session_state["session_id"] = self._session_id
        logger.info("Database session started: %d", self._session_id)

        # Look up known user name from previous sessions
        self._known_user_name = self._db.get_most_recent_user_name()
        if self._known_user_name:
            logger.info("Known user from previous session: '%s'", self._known_user_name)

        # Load today's curriculum plan (only for english_teacher profile)
        from reachy_mini_teacher_app.config import config as _cfg
        if _cfg.REACHY_MINI_CUSTOM_PROFILE == "english_teacher":
            import datetime as _dt
            self._today_date = _dt.date.today().isoformat()
            self._daily_plan = self._db.get_or_create_daily_plan(self._today_date)
            self._db.increment_daily_session_count(self._today_date)
            logger.info(
                "Daily plan for %s: unit %d (%s) — session #%d",
                self._today_date,
                self._daily_plan["unit_id"],
                self._daily_plan["unit_name"],
                self._daily_plan["session_count"] + 1,
            )

        head_wobbler = self.deps.head_wobbler
        if head_wobbler is not None:
            head_wobbler.start()

        # Start tool manager with callback to send results back to Gemini
        self.tool_manager.start_up(tool_callbacks=[self._on_tool_complete])

        reconnect_delay = 2.0  # seconds between reconnect attempts
        while not self._shutdown_requested:
            # (Re)build config from current profile on every reconnect
            _fast_reconnect = False
            if self._profile_switch_event.is_set():
                self._profile_switch_event.clear()
                reinitialize_tools()
                # Clear resumption token — new profile means new system prompt
                self._resumption_token = None
                self._is_first_connection = True
                self._needs_config_rebuild = True
                self._profile_just_switched = True
                _fast_reconnect = True  # skip the 2 s wait after a deliberate switch
                # Start a fresh DB session so old transcript doesn't bleed into new persona
                if self._session_id is not None:
                    self._db.end_session(self._session_id, summary="Profile switched")
                self._session_id = self._db.start_session()
                self._session_state["session_id"] = self._session_id
                self._daily_plan = None
                self._today_date = None
                self._known_user_name = self._db.get_most_recent_user_name()
                logger.info("Profile switch detected — new DB session %d, resumption token cleared", self._session_id)

            live_config = self._build_live_config(gtypes)

            try:
                async with client.aio.live.connect(model=config.GEMINI_MODEL, config=live_config) as session:
                    self._session = session
                    if self._is_first_connection:
                        logger.info("Gemini Live connected (model=%s) — new session", config.GEMINI_MODEL)
                        self._is_first_connection = False
                    else:
                        logger.info("Gemini Live connected (model=%s) — resuming session", config.GEMINI_MODEL)

                    # After a profile switch: send a user turn to break the AI's silence
                    # so the new persona greets the user immediately.
                    if self._profile_just_switched:
                        self._profile_just_switched = False
                        try:
                            await session.send_client_content(
                                turns={"role": "user", "parts": [{"text": "سلام"}]},
                                turn_complete=True,
                            )
                            logger.info("Sent profile-switch kickoff turn to trigger AI greeting")
                        except Exception as _e:
                            logger.debug("Kickoff send_client_content failed: %s", _e)

                    send_task = asyncio.create_task(self._send_loop(session))
                    recv_task = asyncio.create_task(self._recv_loop(session))
                    try:
                        done, pending = await asyncio.wait(
                            [send_task, recv_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for t in done:
                            if t == send_task:
                                logger.warning("_send_loop completed first — triggering reconnect")
                            elif t == recv_task:
                                logger.info("_recv_loop completed — triggering reconnect")
                        for t in pending:
                            t.cancel()
                            try:
                                await t
                            except (asyncio.CancelledError, Exception):
                                pass
                        # Re-raise if shutdown was requested via CancelledError
                        for t in done:
                            if t.cancelled():
                                raise asyncio.CancelledError()
                    except asyncio.CancelledError:
                        send_task.cancel()
                        recv_task.cancel()
                        break
                    finally:
                        # Always discard the resumption token after a session closes.
                        # Session resumption restores the AI to the exact moment it was
                        # speaking — BEFORE the user's last response — so it re-asks the
                        # same question on reconnect. Clearing the token forces a full
                        # transcript rebuild, giving the AI the real conversation state.
                        self._resumption_token = None
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Gemini Live session error: %s", e)
                self._resumption_token = None

            if self._shutdown_requested:
                break

            # Drain stale mic frames so the new session starts clean
            while not self._audio_in_queue.empty():
                try:
                    self._audio_in_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            delay = 0.3 if _fast_reconnect else reconnect_delay
            logger.info("Reconnecting to Gemini Live in %.1fs …", delay)
            await asyncio.sleep(delay)

        self._session = None
        if head_wobbler is not None:
            head_wobbler.stop()

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    async def _send_loop(self, session: Any) -> None:
        """Continuously read buffered mic frames and forward them to Gemini."""
        from google.genai import types as gtypes

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

            # Skip invalid frames
            if input_sr == 0 or len(audio_frame) == 0:
                continue

            # Flatten stereo → mono
            if audio_frame.ndim == 2:
                if audio_frame.shape[1] > audio_frame.shape[0]:
                    audio_frame = audio_frame.T
                if audio_frame.shape[1] > 1:
                    audio_frame = audio_frame[:, 0]

            # Resample to 16 kHz if needed
            if input_sr != GEMINI_INPUT_SAMPLE_RATE:
                num_samples = int(len(audio_frame) * GEMINI_INPUT_SAMPLE_RATE / input_sr)
                if num_samples > 0:
                    audio_frame = resample(audio_frame, num_samples)
                else:
                    continue

            pcm_int16 = audio_to_int16(audio_frame)
            try:
                # Use session.send_realtime_input(audio=...) — the non-deprecated API.
                # session.send() is deprecated and its _parse_client_message() always
                # serialises to realtime_input.media_chunks regardless of which field
                # you populate in LiveClientRealtimeInput, causing server error 1007.
                await session.send_realtime_input(
                    audio=gtypes.Blob(
                        data=pcm_int16.tobytes(),
                        mime_type="audio/pcm;rate=16000",
                    )
                )
            except Exception as e:
                logger.debug("send_realtime_input error (session closing?): %s", e)

    async def _recv_loop(self, session: Any) -> None:
        """Receive Gemini responses: audio, transcripts, and tool calls."""
        try:
            async for response in session.receive():
                if self._shutdown_requested:
                    break

                # Session resumption token — save for next reconnect
                if getattr(response, "session_resumption_update", None) is not None:
                    sru = response.session_resumption_update
                    token = getattr(sru, "resumption_token", None) or getattr(sru, "new_handle", None)
                    if not token and hasattr(sru, "handle"):
                        token = sru.handle
                    if token:
                        self._resumption_token = token
                        logger.debug("Session resumption token updated")
                    if getattr(sru, "resumable", None) is False:
                        logger.warning("Session marked as non-resumable")
                        self._resumption_token = None

                # GoAway — server will close soon
                if getattr(response, "go_away", None) is not None:
                    time_left = getattr(response.go_away, "timeLeft", "unknown")
                    logger.warning("Received GoAway from Gemini — time left: %s", time_left)

                # Audio output
                if response.data is not None:
                    audio_bytes = response.data
                    audio_arr = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
                    # Feed head wobbler
                    hw = self.deps.head_wobbler
                    if hw is not None:
                        hw.feed(base64.b64encode(audio_bytes).decode())
                    await self.output_queue.put((GEMINI_OUTPUT_SAMPLE_RATE, audio_arr))

                # Text / transcript (only check when no audio data)
                elif response.text is not None:
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "assistant", "content": response.text})
                    )
                    # Log assistant text to DB
                    if self._session_id is not None:
                        self._db.add_message(self._session_id, "assistant", response.text)

                # Server-side Content (transcripts, tool calls)
                if response.server_content is not None:
                    sc = response.server_content

                    # User speech transcript
                    if hasattr(sc, "input_transcription") and sc.input_transcription:
                        txt = sc.input_transcription.text
                        if txt:
                            await self.output_queue.put(
                                AdditionalOutputs({"role": "user", "content": txt})
                            )
                            # Log user speech to DB
                            if self._session_id is not None:
                                self._db.add_message(self._session_id, "user", txt)

                    # Assistant speech transcript (output transcription)
                    if hasattr(sc, "output_transcription") and sc.output_transcription:
                        txt = sc.output_transcription.text
                        if txt:
                            # Log assistant speech to DB (for session continuity)
                            if self._session_id is not None:
                                self._db.add_message(self._session_id, "assistant", txt)

                    # User interrupted the AI — flush pending audio
                    if getattr(sc, "interrupted", False):
                        logger.info("User interrupted AI — flushing audio queue")
                        if self._clear_queue is not None:
                            self._clear_queue()

                    # Turn complete
                    if getattr(sc, "turn_complete", False):
                        hw = self.deps.head_wobbler
                        if hw is not None:
                            hw.reset()
                        mm = self.deps.movement_manager
                        if mm is not None:
                            mm.mark_activity()

                # Tool calls
                if response.tool_call is not None:
                    for fc in response.tool_call.function_calls:
                        tool_name = fc.name
                        args_dict: Dict[str, Any] = dict(fc.args) if fc.args else {}
                        call_id: str = str(fc.id)

                        logger.info("Tool call: %s args=%s", tool_name, args_dict)

                        bg_tool = await self.tool_manager.start_tool(
                            call_id=call_id,
                            tool_call_routine=ToolCallRoutine(
                                tool_name=tool_name,
                                args_json_str=json.dumps(args_dict),
                                deps=self.deps,
                            ),
                            is_idle_tool_call=False,
                        )

                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "role": "assistant",
                                    "content": f"🛠️ Used tool {tool_name}. ID: {bg_tool.tool_id}",
                                }
                            )
                        )

        except asyncio.CancelledError:
            logger.debug("_recv_loop cancelled")
        except Exception as e:
            logger.error("_recv_loop error: %s", e)
        else:
            logger.warning("_recv_loop: session.receive() iterator ended (session closed by server)")

    async def _on_tool_complete(self, notification: ToolNotification) -> None:
        """Send FunctionResponse back to Gemini when a tool finishes."""
        from google.genai import types as gtypes

        session = self._session
        if session is None:
            logger.warning("Tool %s completed but no active Gemini session to send response to", notification.tool_name)
            return

        # Build the response payload
        if notification.error:
            response_dict = {"error": notification.error}
        elif notification.result:
            # Sanitize: strip large binary fields (e.g. b64 images) that
            # the Gemini Live FunctionResponse cannot handle.
            response_dict = {
                k: v for k, v in notification.result.items()
                if not (isinstance(v, str) and len(v) > 10_000)
            }
            if not response_dict:
                response_dict = {"result": "ok"}
        else:
            response_dict = {"result": "ok"}

        function_response = gtypes.FunctionResponse(
            id=notification.id,
            name=notification.tool_name,
            response=response_dict,
        )

        try:
            await session.send_tool_response(function_responses=[function_response])
            logger.info("Sent tool response for %s (id=%s)", notification.tool_name, notification.id)
        except Exception as e:
            logger.error("Failed to send tool response for %s: %s", notification.tool_name, e)
