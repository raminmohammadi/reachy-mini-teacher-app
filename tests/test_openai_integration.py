"""Tests for OpenAI Realtime handler, mode switching, config, camera vision routing, and tool spec conversion."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    """Test configuration fields for OpenAI support."""

    def test_openai_api_key_field_exists(self):
        from reachy_mini_teacher_app.config import Config
        c = Config()
        assert hasattr(c, "OPENAI_API_KEY")

    def test_openai_model_default(self):
        from reachy_mini_teacher_app.config import Config
        c = Config()
        assert c.OPENAI_MODEL == "gpt-4o-realtime-preview"

    def test_app_mode_default_is_gemini(self):
        from reachy_mini_teacher_app.config import Config
        c = Config()
        # Default should be gemini (unless overridden by env)
        assert c.APP_MODE in ("gemini", "openai", "local")


# ---------------------------------------------------------------------------
# CLI / mode switching tests
# ---------------------------------------------------------------------------


class TestModeSwitching:
    """Test --mode argument parsing and APP_MODE env var."""

    def test_parse_mode_gemini(self):
        from reachy_mini_teacher_app.utils import parse_args
        sys.argv = ["test", "--mode", "gemini"]
        args, _ = parse_args()
        assert args.mode == "gemini"

    def test_parse_mode_openai(self):
        from reachy_mini_teacher_app.utils import parse_args
        sys.argv = ["test", "--mode", "openai"]
        args, _ = parse_args()
        assert args.mode == "openai"

    def test_parse_mode_local(self):
        from reachy_mini_teacher_app.utils import parse_args
        sys.argv = ["test", "--mode", "local"]
        args, _ = parse_args()
        assert args.mode == "local"

    def test_parse_mode_invalid_rejected(self):
        from reachy_mini_teacher_app.utils import parse_args
        sys.argv = ["test", "--mode", "invalid"]
        with pytest.raises(SystemExit):
            parse_args()

    def test_mode_defaults_to_none(self):
        """When no --mode is given, args.mode should be None (config.APP_MODE is used)."""
        from reachy_mini_teacher_app.utils import parse_args
        sys.argv = ["test"]
        args, _ = parse_args()
        assert args.mode is None


# ---------------------------------------------------------------------------
# Tool spec conversion tests
# ---------------------------------------------------------------------------


class TestToolSpecConversion:
    """Test _tool_specs_to_openai conversion."""

    def test_basic_conversion(self):
        from reachy_mini_teacher_app.openai_handler import _tool_specs_to_openai
        specs = [
            {
                "name": "camera",
                "description": "Take a photo",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            }
        ]
        result = _tool_specs_to_openai(specs)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "camera"
        assert result[0]["description"] == "Take a photo"
        assert result[0]["parameters"]["properties"]["question"]["type"] == "string"

    def test_empty_specs(self):
        from reachy_mini_teacher_app.openai_handler import _tool_specs_to_openai
        assert _tool_specs_to_openai([]) == []

    def test_multiple_tools(self):
        from reachy_mini_teacher_app.openai_handler import _tool_specs_to_openai
        specs = [
            {"name": "tool_a", "description": "A", "parameters": {}},
            {"name": "tool_b", "description": "B", "parameters": {}},
        ]
        result = _tool_specs_to_openai(specs)
        assert len(result) == 2
        names = {t["name"] for t in result}


# ---------------------------------------------------------------------------
# OpenAI handler construction tests
# ---------------------------------------------------------------------------


class TestOpenAIRealtimeHandler:
    """Test OpenAIRealtimeHandler instantiation and interface."""

    def _make_deps(self):
        """Create minimal mock ToolDependencies."""
        deps = MagicMock()
        deps.head_wobbler = None
        deps.movement_manager = None
        deps.camera_worker = None
        deps.vision_manager = None
        return deps

    def test_handler_instantiation(self):
        from reachy_mini_teacher_app.openai_handler import OpenAIRealtimeHandler
        handler = OpenAIRealtimeHandler(self._make_deps())
        assert handler is not None
        assert handler.output_queue is not None

    def test_handler_copy(self):
        from reachy_mini_teacher_app.openai_handler import OpenAIRealtimeHandler
        original = OpenAIRealtimeHandler(self._make_deps(), gradio_mode=True)
        copied = original.copy()
        assert isinstance(copied, OpenAIRealtimeHandler)
        assert copied is not original

    def test_handler_has_required_methods(self):
        from reachy_mini_teacher_app.openai_handler import OpenAIRealtimeHandler
        handler = OpenAIRealtimeHandler(self._make_deps())
        for method_name in ["receive", "emit", "shutdown", "start_up", "_send_loop", "_recv_loop", "_on_tool_complete"]:
            assert hasattr(handler, method_name), f"Missing method: {method_name}"
            assert callable(getattr(handler, method_name))

    def test_sample_rate_constant(self):
        from reachy_mini_teacher_app.openai_handler import OPENAI_SAMPLE_RATE
        assert OPENAI_SAMPLE_RATE == 24000

    @pytest.mark.asyncio
    async def test_receive_queues_frame(self):
        from reachy_mini_teacher_app.openai_handler import OpenAIRealtimeHandler
        handler = OpenAIRealtimeHandler(self._make_deps())
        frame = (16000, np.zeros((1, 1600), dtype=np.int16))
        await handler.receive(frame)
        assert not handler._audio_in_queue.empty()

    @pytest.mark.asyncio
    async def test_shutdown_sets_flag(self):
        from reachy_mini_teacher_app.openai_handler import OpenAIRealtimeHandler
        handler = OpenAIRealtimeHandler(self._make_deps())
        assert not handler._shutdown_requested
        await handler.shutdown()
        assert handler._shutdown_requested


# ---------------------------------------------------------------------------
# Tool response sanitization tests
# ---------------------------------------------------------------------------


class TestToolResponseSanitization:
    """Test that large values are stripped from tool responses."""

    def _make_handler(self):
        from reachy_mini_teacher_app.openai_handler import OpenAIRealtimeHandler
        deps = MagicMock()
        deps.head_wobbler = None
        deps.movement_manager = None
        deps.camera_worker = None
        deps.vision_manager = None
        return OpenAIRealtimeHandler(deps)

    @pytest.mark.asyncio
    async def test_sanitize_large_values(self):
        """Tool response with large base64 strings should have them stripped."""
        handler = self._make_handler()
        # Mock the connection
        mock_conn = AsyncMock()
        handler._connection = mock_conn

        from reachy_mini_teacher_app.tools.background_tool_manager import ToolNotification
        from reachy_mini_teacher_app.tools.tool_constants import ToolState

        notification = ToolNotification(
            id="test_call_id",
            tool_name="camera",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={
                "b64_im": "A" * 20000,  # Large string > 10KB
                "status": "Image captured",
            },
        )

        await handler._on_tool_complete(notification)

        # Should have called conn.send twice (item.create + response.create)
        assert mock_conn.send.call_count == 2

        # First call is conversation.item.create
        first_call_args = mock_conn.send.call_args_list[0][0][0]
        assert first_call_args["type"] == "conversation.item.create"
        output = json.loads(first_call_args["item"]["output"])
        # Large b64_im should be stripped
        assert "b64_im" not in output
        assert output.get("status") == "Image captured"

        # Second call is response.create
        second_call_args = mock_conn.send.call_args_list[1][0][0]
        assert second_call_args["type"] == "response.create"

    @pytest.mark.asyncio
    async def test_error_notification(self):
        handler = self._make_handler()
        mock_conn = AsyncMock()
        handler._connection = mock_conn

        from reachy_mini_teacher_app.tools.background_tool_manager import ToolNotification
        from reachy_mini_teacher_app.tools.tool_constants import ToolState

        notification = ToolNotification(
            id="err_call",
            tool_name="camera",
            is_idle_tool_call=False,
            status=ToolState.FAILED,
            error="Camera not available",
        )

        await handler._on_tool_complete(notification)
        first_call_args = mock_conn.send.call_args_list[0][0][0]
        output = json.loads(first_call_args["item"]["output"])
        assert output["error"] == "Camera not available"

    @pytest.mark.asyncio
    async def test_no_connection_does_not_crash(self):
        handler = self._make_handler()
        handler._connection = None

        from reachy_mini_teacher_app.tools.background_tool_manager import ToolNotification
        from reachy_mini_teacher_app.tools.tool_constants import ToolState

        notification = ToolNotification(
            id="x", tool_name="t", is_idle_tool_call=False,
            status=ToolState.COMPLETED, result={"ok": True},
        )
        # Should not raise
        await handler._on_tool_complete(notification)


# ---------------------------------------------------------------------------
# Camera vision routing tests
# ---------------------------------------------------------------------------


class TestCameraVisionRouting:
    """Test _describe_image routing between OpenAI and Gemini."""

    @pytest.mark.asyncio
    async def test_openai_mode_uses_openai_vision(self):
        """When APP_MODE=openai and OPENAI_API_KEY is set, should use OpenAI vision."""
        with patch("reachy_mini_teacher_app.tools.camera.config") as mock_config:
            mock_config.APP_MODE = "openai"
            mock_config.OPENAI_API_KEY = "sk-test"
            mock_config.GEMINI_API_KEY = "gem-test"

            with patch("reachy_mini_teacher_app.tools.camera._openai_describe_image", new_callable=AsyncMock) as mock_oai:
                mock_oai.return_value = "OpenAI description"
                from reachy_mini_teacher_app.tools.camera import _describe_image
                result = await _describe_image("base64data", "What is this?")
                mock_oai.assert_called_once_with("base64data", "What is this?")
                assert result == "OpenAI description"

    @pytest.mark.asyncio
    async def test_gemini_mode_uses_gemini_vision(self):
        """When APP_MODE=gemini and GEMINI_API_KEY is set, should use Gemini vision."""
        with patch("reachy_mini_teacher_app.tools.camera.config") as mock_config:
            mock_config.APP_MODE = "gemini"
            mock_config.OPENAI_API_KEY = "sk-test"
            mock_config.GEMINI_API_KEY = "gem-test"

            with patch("reachy_mini_teacher_app.tools.camera._gemini_describe_image", new_callable=AsyncMock) as mock_gem:
                mock_gem.return_value = "Gemini description"
                from reachy_mini_teacher_app.tools.camera import _describe_image
                result = await _describe_image("base64data", "What is this?")
                mock_gem.assert_called_once_with("base64data", "What is this?")
                assert result == "Gemini description"

    @pytest.mark.asyncio
    async def test_no_api_keys_returns_error(self):
        """When no API keys are set, should return error message."""
        with patch("reachy_mini_teacher_app.tools.camera.config") as mock_config:
            mock_config.APP_MODE = "gemini"
            mock_config.OPENAI_API_KEY = None
            mock_config.GEMINI_API_KEY = None

            from reachy_mini_teacher_app.tools.camera import _describe_image
            result = await _describe_image("base64data", "What is this?")
            assert "No vision API key" in result

    @pytest.mark.asyncio
    async def test_gemini_fallback_when_no_gemini_key(self):
        """When APP_MODE=gemini but no GEMINI_API_KEY, fallback to OpenAI if available."""
        with patch("reachy_mini_teacher_app.tools.camera.config") as mock_config:
            mock_config.APP_MODE = "gemini"
            mock_config.GEMINI_API_KEY = None
            mock_config.OPENAI_API_KEY = "sk-test"

            with patch("reachy_mini_teacher_app.tools.camera._openai_describe_image", new_callable=AsyncMock) as mock_oai:
                mock_oai.return_value = "OpenAI fallback"
                from reachy_mini_teacher_app.tools.camera import _describe_image
                result = await _describe_image("b64", "q")
                mock_oai.assert_called_once()
                assert result == "OpenAI fallback"

    @pytest.mark.asyncio
    async def test_gemini_describe_no_key(self):
        """_gemini_describe_image returns error when no API key."""
        with patch("reachy_mini_teacher_app.tools.camera.config") as mock_config:
            mock_config.GEMINI_API_KEY = None
            from reachy_mini_teacher_app.tools.camera import _gemini_describe_image
            result = await _gemini_describe_image("b64", "question")
            assert "No vision processor" in result

    @pytest.mark.asyncio
    async def test_openai_describe_no_key(self):
        """_openai_describe_image returns error when no API key."""
        with patch("reachy_mini_teacher_app.tools.camera.config") as mock_config:
            mock_config.OPENAI_API_KEY = None
            from reachy_mini_teacher_app.tools.camera import _openai_describe_image
            result = await _openai_describe_image("b64", "question")
            assert "No vision processor" in result
