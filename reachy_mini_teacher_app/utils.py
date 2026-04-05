"""Shared utilities for argument parsing, logging, and vision setup."""

import logging
import argparse
import warnings
from typing import Any, Tuple, Optional

from reachy_mini import ReachyMini
from reachy_mini_teacher_app.camera_worker import CameraWorker


def parse_args() -> Tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Local App")
    parser.add_argument(
        "--mode",
        choices=["gemini", "openai", "local"],
        default=None,
        help="AI backend: 'gemini' (Gemini Live API), 'openai' (OpenAI Realtime API), "
             "or 'local' (Whisper + Ollama + Kokoro). "
             "Defaults to APP_MODE env var, then 'gemini'.",
    )
    parser.add_argument(
        "--head-tracker",
        choices=["yolo", "mediapipe", None],
        default=None,
        help="Choose head tracker (default: None)",
    )
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--local-vision",
        default=False,
        action="store_true",
        help="Use local SmolVLM2 vision model for camera tool queries",
    )
    parser.add_argument("--gradio", default=False, action="store_true", help="Open Gradio web interface")
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="[Optional] Robot name/prefix for Zenoh topics",
    )
    return parser.parse_known_args()


def handle_vision_stuff(args: argparse.Namespace, current_robot: ReachyMini) -> Tuple[CameraWorker | None, Any, Any]:
    """Initialize camera worker, head tracker, and optionally local vision manager."""
    camera_worker = None
    head_tracker = None
    vision_manager = None

    if not args.no_camera:
        if args.head_tracker is not None:
            if args.head_tracker == "yolo":
                from reachy_mini_teacher_app.vision.yolo_head_tracker import HeadTracker
                head_tracker = HeadTracker()
            elif args.head_tracker == "mediapipe":
                from reachy_mini_toolbox.vision import HeadTracker  # type: ignore[no-redef]
                head_tracker = HeadTracker()

        camera_worker = CameraWorker(current_robot, head_tracker)

        if args.local_vision:
            try:
                from reachy_mini_teacher_app.vision.processors import initialize_vision_manager
                vision_manager = initialize_vision_manager(camera_worker)
            except ImportError as e:
                raise ImportError(
                    "To use --local-vision, please install the extra dependencies: "
                    "pip install '.[local_vision]'",
                ) from e
        else:
            logging.getLogger(__name__).info(
                "Local vision disabled. Use --local-vision to enable SmolVLM2 processing.",
            )

    return camera_worker, head_tracker, vision_manager


def setup_logger(debug: bool) -> logging.Logger:
    """Set up root logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
    )
    logger = logging.getLogger(__name__)

    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("fastrtc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("fastrtc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)

    return logger


def log_connection_troubleshooting(logger: logging.Logger, robot_name: Optional[str]) -> None:
    """Log troubleshooting steps for connection issues."""
    logger.error("Troubleshooting steps:")
    logger.error("  1. Verify reachy-mini-daemon is running")
    if robot_name is not None:
        logger.error("  2. Daemon must be started with: --robot-name '%s'", robot_name)
    else:
        logger.error("  2. If daemon uses --robot-name, add the same flag here: --robot-name <name>")
    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")

