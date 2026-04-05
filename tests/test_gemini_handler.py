"""Tests for GeminiLiveHandler – construction, tool spec conversion, shutdown."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

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

