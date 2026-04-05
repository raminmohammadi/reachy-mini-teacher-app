"""Tests for BackgroundToolManager – lifecycle, progress, cancel, cleanup."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_teacher_app.tools.background_tool_manager import (
    BackgroundTool,
    BackgroundToolManager,
    ToolCallRoutine,
    ToolNotification,
    ToolProgress,
)
from reachy_mini_teacher_app.tools.tool_constants import ToolState


def _make_deps():
    deps = MagicMock()
    deps.head_wobbler = None
    deps.movement_manager = None
    deps.camera_worker = None
    deps.vision_manager = None
    return deps


# ── ToolProgress ─────────────────────────────────────────────────────────


class TestToolProgress:
    def test_valid(self):
        tp = ToolProgress(progress=0.5, message="halfway")
        assert tp.progress == 0.5

    def test_bounds(self):
        with pytest.raises(Exception):
            ToolProgress(progress=-0.1)
        with pytest.raises(Exception):
            ToolProgress(progress=1.1)


# ── ToolNotification ─────────────────────────────────────────────────────


class TestToolNotification:
    def test_create(self):
        n = ToolNotification(
            id="abc", tool_name="camera", is_idle_tool_call=False,
            status=ToolState.COMPLETED, result={"ok": True},
        )
        assert n.tool_name == "camera"
        assert n.status == ToolState.COMPLETED

    def test_error_notification(self):
        n = ToolNotification(
            id="err", tool_name="camera", is_idle_tool_call=False,
            status=ToolState.FAILED, error="broken",
        )
        assert n.error == "broken"


# ── BackgroundTool ───────────────────────────────────────────────────────


class TestBackgroundTool:
    def test_tool_id(self):
        t = BackgroundTool(
            id="abc", tool_name="camera", is_idle_tool_call=False, status=ToolState.RUNNING,
        )
        assert "camera" in t.tool_id and "abc" in t.tool_id

    def test_get_notification(self):
        t = BackgroundTool(
            id="abc", tool_name="camera", is_idle_tool_call=False,
            status=ToolState.COMPLETED, result={"ok": True},
        )
        n = t.get_notification()
        assert isinstance(n, ToolNotification)
        assert n.result == {"ok": True}


# ── BackgroundToolManager ───────────────────────────────────────────────


class TestBackgroundToolManager:
    @pytest.mark.asyncio
    async def test_start_and_complete_tool(self):
        mgr = BackgroundToolManager()
        callback = AsyncMock()
        mgr.start_up(tool_callbacks=[callback])

        routine = MagicMock(spec=ToolCallRoutine)
        routine.tool_name = "test_tool"
        routine.__call__ = AsyncMock(return_value={"result": "done"})

        bg = await mgr.start_tool(call_id="c1", tool_call_routine=routine, is_idle_tool_call=False)
        assert bg.status == ToolState.RUNNING

        # Wait for completion
        await asyncio.sleep(0.2)
        await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_cancel_tool(self):
        mgr = BackgroundToolManager()
        callback = AsyncMock()
        mgr.start_up(tool_callbacks=[callback])

        async def slow_tool(manager):
            await asyncio.sleep(10)
            return {"result": "done"}

        routine = MagicMock(spec=ToolCallRoutine)
        routine.tool_name = "slow"
        routine.__call__ = slow_tool

        bg = await mgr.start_tool(call_id="c2", tool_call_routine=routine, is_idle_tool_call=False)
        cancelled = await mgr.cancel_tool(bg.tool_id)
        assert cancelled is True
        await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self):
        mgr = BackgroundToolManager()
        mgr.set_loop()
        assert await mgr.cancel_tool("nonexistent") is False

    @pytest.mark.asyncio
    async def test_get_running_tools(self):
        mgr = BackgroundToolManager()
        callback = AsyncMock()
        mgr.start_up(tool_callbacks=[callback])

        async def slow(manager):
            await asyncio.sleep(10)
            return {}

        routine = MagicMock(spec=ToolCallRoutine)
        routine.tool_name = "slow"
        routine.__call__ = slow

        await mgr.start_tool(call_id="r1", tool_call_routine=routine, is_idle_tool_call=False)
        running = mgr.get_running_tools()
        assert len(running) >= 1
        await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_update_progress(self):
        mgr = BackgroundToolManager()
        callback = AsyncMock()
        mgr.start_up(tool_callbacks=[callback])

        async def slow(manager):
            await asyncio.sleep(10)
            return {}

        routine = MagicMock(spec=ToolCallRoutine)
        routine.tool_name = "prog"
        routine.__call__ = slow

        bg = await mgr.start_tool(call_id="p1", tool_call_routine=routine, is_idle_tool_call=False, with_progress=True)
        updated = await mgr.update_progress(bg.tool_id, 0.5, "half done")
        assert updated is True
        assert mgr.get_tool(bg.tool_id).progress.progress == 0.5
        await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self):
        mgr = BackgroundToolManager()
        mgr.start_up(tool_callbacks=[AsyncMock()])
        await mgr.shutdown()
        # Lifecycle tasks should be cleared
        assert len(mgr._lifecycle_tasks) == 0

