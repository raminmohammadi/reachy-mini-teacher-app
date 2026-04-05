"""Entry point for the Reachy Mini Local App.

Usage examples
──────────────
# Gemini Live mode (default):
reachy-mini-local-app --mode gemini

# Fully local mode (Whisper + Ollama + Kokoro):
reachy-mini-local-app --mode local

# With camera and local SmolVLM2 vision:
reachy-mini-local-app --mode local --local-vision

# Open Gradio web UI instead of console:
reachy-mini-local-app --mode gemini --gradio
"""

from __future__ import annotations

import datetime
import signal
import sys
import logging
import threading
from pathlib import Path

# reachy_mini is only available on the physical robot.
# Both imports are guarded so the module (and therefore the package) can be
# imported safely in CI and on developer machines without robot hardware.
try:
    from reachy_mini import ReachyMini  # type: ignore[import-untyped]
except ImportError:
    ReachyMini = None  # type: ignore[assignment,misc]

try:
    from reachy_mini import ReachyMiniApp as _ReachyMiniAppBase  # type: ignore[attr-defined]
except ImportError:
    _ReachyMiniAppBase = object  # type: ignore[assignment,misc]

from reachy_mini_teacher_app.config import config
from reachy_mini_teacher_app.utils import parse_args, setup_logger, handle_vision_stuff, log_connection_troubleshooting
from reachy_mini_teacher_app.moves import MovementManager
from reachy_mini_teacher_app.audio.head_wobbler import HeadWobbler
from reachy_mini_teacher_app.tools.core_tools import ToolDependencies
from reachy_mini_teacher_app.console import LocalStream


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Web UI constants
# ---------------------------------------------------------------------------

_WEB_UI_PORT = 7860
_STATIC_DIR = Path(__file__).parent / "static"

# Shared state written by _run(), read by FastAPI endpoints.
_app_state: dict = {
    "handler": None,
    "mode": "unknown",
    "start_time": None,
}

# ---------------------------------------------------------------------------
# Reachy Mini Apps integration — HuggingFace Space compatible
# ---------------------------------------------------------------------------

class ReachyMiniTeacherApp(_ReachyMiniAppBase):  # type: ignore[valid-type,misc]
    """Entry-point class compatible with the Reachy Mini Apps dashboard.

    The dashboard (reachy-mini-app-assistant) discovers this class through
    the package's ``__init__.py``, then calls ``run()`` in a managed thread,
    passing a ``stop_event`` it sets when the user stops the app.
    """

    # Dashboard link — shown as a clickable URL in the HF Space UI.
    custom_app_url: str | None = f"http://localhost:{_WEB_UI_PORT}"

    # ------------------------------------------------------------------
    # HuggingFace / dashboard interface
    # ------------------------------------------------------------------

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Start the app; blocks until *stop_event* is set.

        Called by the Reachy Mini Apps dashboard.  The method must not return
        until the app has fully shut down.
        """
        _start_web_ui(stop_event)
        _run(reachy_mini, gradio_mode=False, stop_event=stop_event)

    # ------------------------------------------------------------------
    # Legacy CLI / plugin hook (kept for backward compatibility)
    # ------------------------------------------------------------------

    def start(
        self,
        robot: ReachyMini,
        settings_app: object = None,
        instance_path: str | None = None,
    ) -> None:
        _run(robot, gradio_mode=False)

    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run(
    robot: ReachyMini,
    *,
    gradio_mode: bool = False,
    args=None,
    stop_event: threading.Event | None = None,
) -> None:
    """Initialise subsystems and start the chosen handler.

    Blocks until the stream closes.  When *stop_event* is provided (HF
    dashboard mode) a background watcher thread closes the stream as soon as
    the event fires, so ``run()`` returns cleanly.
    """

    # --- Movement Manager ---
    movement_manager = MovementManager(robot)
    movement_manager.start()

    # --- Vision / Camera ---
    camera_worker, _head_tracker, vision_manager = handle_vision_stuff(args or _dummy_args(), robot)
    if camera_worker is not None:
        camera_worker.start()

    # --- Head Wobbler ---
    def _set_speech_offsets(offsets):
        movement_manager.set_speech_offsets(offsets)

    head_wobbler = HeadWobbler(set_speech_offsets=_set_speech_offsets)

    # --- Tool Dependencies ---
    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )

    # --- Select handler ---
    mode = config.APP_MODE
    if args is not None and getattr(args, "mode", None):
        mode = args.mode

    if mode == "local":
        from reachy_mini_teacher_app.local_handler import LocalPipelineHandler
        handler = LocalPipelineHandler(deps, gradio_mode=gradio_mode)
        logger.info("Starting in LOCAL mode (Whisper + Ollama + Kokoro)")
    elif mode == "openai":
        from reachy_mini_teacher_app.openai_handler import OpenAIRealtimeHandler
        handler = OpenAIRealtimeHandler(deps, gradio_mode=gradio_mode)
        logger.info("Starting in OPENAI REALTIME mode")
    else:
        from reachy_mini_teacher_app.gemini_handler import GeminiLiveHandler
        handler = GeminiLiveHandler(deps, gradio_mode=gradio_mode)
        logger.info("Starting in GEMINI LIVE mode")

    stream = LocalStream(handler=handler, robot=robot)

    # --- Populate shared state so the web UI API can read it ---
    _app_state["handler"] = handler
    _app_state["mode"] = mode
    _app_state["start_time"] = datetime.datetime.now().isoformat()

    # --- Unified shutdown helper ---
    def _do_shutdown() -> None:
        stream.close()
        movement_manager.stop()
        if camera_worker is not None:
            camera_worker.stop()

    # --- Graceful shutdown on Ctrl-C / SIGTERM (CLI mode) ---
    def _signal_shutdown(sig, frame) -> None:
        logger.info("Interrupt received; shutting down …")
        _do_shutdown()

    signal.signal(signal.SIGINT, _signal_shutdown)
    signal.signal(signal.SIGTERM, _signal_shutdown)

    # --- HF dashboard stop_event watcher ---
    if stop_event is not None:
        def _watch_stop_event() -> None:
            stop_event.wait()
            logger.info("Dashboard stop_event fired; shutting down …")
            _do_shutdown()

        threading.Thread(target=_watch_stop_event, daemon=True, name="hf-stop-watcher").start()

    try:
        stream.launch()
    finally:
        movement_manager.stop()
        if camera_worker is not None:
            camera_worker.stop()


class _dummy_args:
    """Minimal args object when running from the plugin system (no CLI)."""
    no_camera = False
    local_vision = False
    head_tracker = None


# ---------------------------------------------------------------------------
# Web UI — FastAPI + uvicorn
# ---------------------------------------------------------------------------

def _get_available_profiles() -> list:
    """Return ['default', ...other profile names...] (no duplicates, no __pycache__)."""
    try:
        extras = sorted(
            p.name
            for p in config.PROFILES_DIRECTORY.iterdir()
            if p.is_dir() and not p.name.startswith("_") and p.name != "default"
        )
        return ["default"] + extras
    except Exception:
        return ["default"]


def _make_fastapi_app():
    """Build the FastAPI application with API endpoints and static file serving."""
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Reachy Mini Local App")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    def get_status():
        handler = _app_state.get("handler")
        session_id = getattr(handler, "_session_id", None) if handler else None
        msg_count = 0
        if handler and session_id is not None:
            try:
                msg_count = len(handler._db.get_session_messages(session_id))
            except Exception:
                pass
        return {
            "mode":          _app_state.get("mode", "unknown"),
            "profile":       config.REACHY_MINI_CUSTOM_PROFILE or "default",
            "session_id":    session_id,
            "message_count": msg_count,
            "start_time":    _app_state.get("start_time"),
            "profiles":      _get_available_profiles(),
        }

    @app.get("/api/messages")
    def get_messages(session_id: int = None, limit: int = 60):  # type: ignore[assignment]
        handler = _app_state.get("handler")
        sid = session_id or (getattr(handler, "_session_id", None) if handler else None)
        if not handler or sid is None:
            return {"messages": [], "session_id": None}
        try:
            msgs = handler._db.get_session_messages(sid)
            return {"messages": msgs[-limit:], "session_id": sid}
        except Exception as e:
            return {"messages": [], "error": str(e), "session_id": sid}

    @app.post("/api/profile/{profile_name}")
    def switch_profile(profile_name: str):
        from reachy_mini_teacher_app.config import set_custom_profile
        available = _get_available_profiles()
        if profile_name not in available:
            return {"ok": False, "error": f"Unknown profile: {profile_name}"}
        set_custom_profile(None if profile_name == "default" else profile_name)
        # Signal the handler to reconnect with the new profile (Gemini mode only)
        handler = _app_state.get("handler")
        if handler is not None:
            ev = getattr(handler, "_profile_switch_event", None)
            if ev is not None:
                try:
                    # Schedule set() on the handler's event loop if available
                    loop = getattr(handler, "_loop", None)
                    if loop and loop.is_running():
                        loop.call_soon_threadsafe(ev.set)
                    else:
                        ev.set()
                except Exception:
                    pass
        return {"ok": True, "profile": profile_name}

    if _STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
    return app


def _start_web_ui(stop_event: threading.Event | None = None) -> None:
    """Start the FastAPI/uvicorn web UI in a background daemon thread."""
    try:
        import uvicorn
    except ImportError:
        logger.warning("uvicorn not installed — web UI disabled (pip install uvicorn)")
        return

    app = _make_fastapi_app()
    server_cfg = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=_WEB_UI_PORT,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(server_cfg)

    def _run_server() -> None:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    t = threading.Thread(target=_run_server, daemon=True, name="webui-server")
    t.start()
    logger.info("Web UI started → http://localhost:%d", _WEB_UI_PORT)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args, _ = parse_args()
    logger = setup_logger(getattr(args, "debug", False))

    robot_name = getattr(args, "robot_name", None)
    try:
        robot = ReachyMini(robot_name=robot_name) if robot_name else ReachyMini()
    except Exception:
        logger.error("Failed to connect to Reachy Mini.")
        log_connection_troubleshooting(logger, robot_name)
        sys.exit(1)

    try:
        _run(robot, gradio_mode=getattr(args, "gradio", False), args=args)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("App stopped.")


if __name__ == "__main__":
    main()

