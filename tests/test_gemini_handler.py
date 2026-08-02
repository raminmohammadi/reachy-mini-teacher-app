"""Tests for GeminiLiveHandler – construction, tool spec conversion, shutdown."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip entire module when heavy signal-processing deps are absent (CI without
# numpy/scipy/fastrtc installed).
np = pytest.importorskip("numpy", reason="numpy not installed")
pytest.importorskip("scipy", reason="scipy not installed")
pytest.importorskip("fastrtc", reason="fastrtc not installed")

from reachy_mini_teacher_app.gemini_handler import (
    GeminiLiveHandler,
    GEMINI_INPUT_SAMPLE_RATE,
    GEMINI_OUTPUT_SAMPLE_RATE,
    _tool_specs_to_gemini,
    _type_str_to_gemini,
)


def _make_deps():
    deps = MagicMock()
    deps.head_wobbler = None
    deps.movement_manager = None
    deps.camera_worker = None
    deps.vision_manager = None
    return deps


# ── Constants ────────────────────────────────────────────────────────────


class TestConstants:
    def test_input_sample_rate(self):
        assert GEMINI_INPUT_SAMPLE_RATE == 16000

    def test_output_sample_rate(self):
        assert GEMINI_OUTPUT_SAMPLE_RATE == 24000


# ── Handler construction ────────────────────────────────────────────────


class TestGeminiLiveHandlerConstruction:
    def test_instantiation(self):
        handler = GeminiLiveHandler(_make_deps())
        assert handler is not None
        assert handler.output_queue is not None

    def test_copy(self):
        original = GeminiLiveHandler(_make_deps(), gradio_mode=True)
        copied = original.copy()
        assert isinstance(copied, GeminiLiveHandler)
        assert copied is not original

    def test_required_methods(self):
        handler = GeminiLiveHandler(_make_deps())
        for method in ["receive", "emit", "shutdown", "start_up", "_send_loop", "_recv_loop"]:
            assert hasattr(handler, method) and callable(getattr(handler, method))

    def test_wake_sleep_methods_removed(self):
        # The voice wake/sleep state machine was removed; the desktop
        # launcher/stop scripts drive the robot's wake/sleep transitions.
        # These methods must not be re-introduced.
        handler = GeminiLiveHandler(_make_deps())
        for removed in (
            "_wake_up", "_go_to_sleep", "_delayed_sleep", "_set_initial_pose",
            "_sleep_timeout_watcher", "_check_wake_word", "_run_wake_word_detection",
        ):
            assert not hasattr(handler, removed), f"Removed method is back: {removed}"
        for removed_attr in ("_is_sleeping", "_wake_word_model"):
            assert not hasattr(handler, removed_attr), f"Removed attr is back: {removed_attr}"

    @pytest.mark.asyncio
    async def test_receive_queues_frame(self):
        handler = GeminiLiveHandler(_make_deps())
        frame = (16000, np.zeros((1, 1600), dtype=np.int16))
        await handler.receive(frame)
        assert not handler._audio_in_queue.empty()

    @pytest.mark.asyncio
    async def test_shutdown_sets_flag(self):
        handler = GeminiLiveHandler(_make_deps())
        assert not handler._shutdown_requested
        await handler.shutdown()
        assert handler._shutdown_requested


# ── Transcript buffering ────────────────────────────────────────────────


class TestTranscriptBuffering:
    """Gemini Live streams transcripts in tiny fragments (often one word).
    The handler must buffer them and only persist one merged DB message
    per turn — otherwise the dashboard transcript floods with dozens of
    one-word bubbles."""

    def _make_handler(self):
        handler = GeminiLiveHandler(_make_deps())
        handler._session_id = 42  # pretend a DB session is active
        handler._db = MagicMock()
        return handler

    def test_buffers_initialised_empty(self):
        handler = GeminiLiveHandler(_make_deps())
        assert handler._user_transcript_buffer == []
        assert handler._assistant_transcript_buffer == []

    def test_flush_combines_assistant_fragments(self):
        handler = self._make_handler()
        for chunk in ("Hello ", "Bob ", "jan", "!"):
            handler._assistant_transcript_buffer.append(chunk)
        handler._flush_transcript_buffers()
        handler._db.add_message.assert_called_once_with(42, "assistant", "Hello Bob jan!")
        assert handler._assistant_transcript_buffer == []

    def test_flush_combines_user_fragments(self):
        handler = self._make_handler()
        for chunk in ("Qual ", "é o ", "nome", "?"):
            handler._user_transcript_buffer.append(chunk)
        handler._flush_transcript_buffers()
        handler._db.add_message.assert_called_once_with(42, "user", "Qual é o nome?")

    def test_flush_writes_both_roles_when_present(self):
        handler = self._make_handler()
        handler._user_transcript_buffer.extend(["hi"])
        handler._assistant_transcript_buffer.extend(["hello ", "back"])
        handler._flush_transcript_buffers()
        # Order: user first, then assistant
        calls = handler._db.add_message.call_args_list
        assert len(calls) == 2
        assert calls[0].args == (42, "user", "hi")
        assert calls[1].args == (42, "assistant", "hello back")

    def test_flush_with_empty_buffers_does_nothing(self):
        handler = self._make_handler()
        handler._flush_transcript_buffers()
        handler._db.add_message.assert_not_called()

    def test_flush_skips_whitespace_only_chunks(self):
        handler = self._make_handler()
        handler._assistant_transcript_buffer.extend(["  ", "\n", " "])
        handler._flush_transcript_buffers()
        handler._db.add_message.assert_not_called()
        assert handler._assistant_transcript_buffer == []

    def test_flush_without_session_id_clears_buffers(self):
        handler = GeminiLiveHandler(_make_deps())
        handler._session_id = None
        handler._db = MagicMock()
        handler._assistant_transcript_buffer.append("orphan text")
        handler._flush_transcript_buffers()
        handler._db.add_message.assert_not_called()
        assert handler._assistant_transcript_buffer == []

    @pytest.mark.asyncio
    async def test_shutdown_flushes_pending_buffers(self):
        handler = self._make_handler()
        handler._assistant_transcript_buffer.append("final partial turn")
        # _generate_session_summary touches the network — stub it out.
        handler._generate_session_summary = AsyncMock(return_value=None)
        await handler.shutdown()
        # The pending fragment must hit the DB before end_session is called
        # so the summary generator sees the full conversation.
        handler._db.add_message.assert_any_call(42, "assistant", "final partial turn")


# ── Tool spec conversion ────────────────────────────────────────────────


class TestToolSpecConversion:
    def _get_gtypes(self):
        from google.genai import types as gtypes
        return gtypes

    def test_type_str_mapping(self):
        gtypes = self._get_gtypes()
        assert _type_str_to_gemini("string", gtypes) == gtypes.Type.STRING
        assert _type_str_to_gemini("number", gtypes) == gtypes.Type.NUMBER
        assert _type_str_to_gemini("integer", gtypes) == gtypes.Type.INTEGER
        assert _type_str_to_gemini("boolean", gtypes) == gtypes.Type.BOOLEAN
        assert _type_str_to_gemini("array", gtypes) == gtypes.Type.ARRAY
        assert _type_str_to_gemini("object", gtypes) == gtypes.Type.OBJECT

    def test_unknown_type_defaults_to_string(self):
        gtypes = self._get_gtypes()
        assert _type_str_to_gemini("foobar", gtypes) == gtypes.Type.STRING

    def test_basic_conversion(self):
        gtypes = self._get_gtypes()
        specs = [{
            "name": "camera",
            "description": "Take a photo",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "What to ask"}},
                "required": ["question"],
            },
        }]
        result = _tool_specs_to_gemini(specs, gtypes)
        assert len(result) == 1
        assert result[0].name == "camera"

    def test_empty_specs(self):
        gtypes = self._get_gtypes()
        assert _tool_specs_to_gemini([], gtypes) == []

    def test_no_parameters(self):
        gtypes = self._get_gtypes()
        specs = [{"name": "ping", "description": "Ping", "parameters": {}}]
        result = _tool_specs_to_gemini(specs, gtypes)
        assert len(result) == 1


# ── Tool response ────────────────────────────────────────────────────────


class TestToolResponse:
    @pytest.mark.asyncio
    async def test_no_session_does_not_crash(self):
        handler = GeminiLiveHandler(_make_deps())
        handler._session = None
        from reachy_mini_teacher_app.tools.background_tool_manager import ToolNotification
        from reachy_mini_teacher_app.tools.tool_constants import ToolState
        notif = ToolNotification(
            id="x", tool_name="t", is_idle_tool_call=False,
            status=ToolState.COMPLETED, result={"ok": True},
        )
        await handler._on_tool_complete(notif)  # should not raise

