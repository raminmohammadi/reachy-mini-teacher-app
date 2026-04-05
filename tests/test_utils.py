"""Tests for utils.py – argument parsing, logger setup."""
from __future__ import annotations

import sys
import logging
from unittest.mock import MagicMock

import pytest

from reachy_mini_teacher_app.utils import parse_args, setup_logger


# ── parse_args ───────────────────────────────────────────────────────────


class TestParseArgs:
    def test_default_mode_none(self):
        sys.argv = ["test"]
        args, _ = parse_args()
        assert args.mode is None

    def test_mode_gemini(self):
        sys.argv = ["test", "--mode", "gemini"]
        args, _ = parse_args()
        assert args.mode == "gemini"

    def test_mode_openai(self):
        sys.argv = ["test", "--mode", "openai"]
        args, _ = parse_args()
        assert args.mode == "openai"

    def test_mode_local(self):
        sys.argv = ["test", "--mode", "local"]
        args, _ = parse_args()
        assert args.mode == "local"

    def test_invalid_mode(self):
        sys.argv = ["test", "--mode", "invalid"]
        with pytest.raises(SystemExit):
            parse_args()

    def test_no_camera_flag(self):
        sys.argv = ["test", "--no-camera"]
        args, _ = parse_args()
        assert args.no_camera is True

    def test_debug_flag(self):
        sys.argv = ["test", "--debug"]
        args, _ = parse_args()
        assert args.debug is True

    def test_gradio_flag(self):
        sys.argv = ["test", "--gradio"]
        args, _ = parse_args()
        assert args.gradio is True

    def test_robot_name(self):
        sys.argv = ["test", "--robot-name", "reachy1"]
        args, _ = parse_args()
        assert args.robot_name == "reachy1"

    def test_head_tracker_yolo(self):
        sys.argv = ["test", "--head-tracker", "yolo"]
        args, _ = parse_args()
        assert args.head_tracker == "yolo"

    def test_local_vision_flag(self):
        sys.argv = ["test", "--local-vision"]
        args, _ = parse_args()
        assert args.local_vision is True


# ── setup_logger ─────────────────────────────────────────────────────────


class TestSetupLogger:
    def test_info_level(self):
        # Reset root logger so basicConfig takes effect
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)
        setup_logger(debug=False)
        assert root.level == logging.INFO

    def test_debug_level(self):
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)
        setup_logger(debug=True)
        assert root.level == logging.DEBUG


# ── Tool constants ───────────────────────────────────────────────────────


class TestToolConstants:
    def test_tool_state_values(self):
        from reachy_mini_teacher_app.tools.tool_constants import ToolState
        assert ToolState.RUNNING.value == "running"
        assert ToolState.COMPLETED.value == "completed"
        assert ToolState.FAILED.value == "failed"
        assert ToolState.CANCELLED.value == "cancelled"

    def test_system_tool_values(self):
        from reachy_mini_teacher_app.tools.tool_constants import SystemTool
        assert SystemTool.TASK_STATUS.value == "task_status"
        assert SystemTool.TASK_CANCEL.value == "task_cancel"

