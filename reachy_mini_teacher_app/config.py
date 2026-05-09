"""Configuration for Reachy Mini Teacher App (Gemini + Local pipeline)."""

import os
import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


LOCKED_PROFILE: str | None = None
DEFAULT_PROFILES_DIRECTORY = Path(__file__).parent / "profiles"

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean value for %s=%r, using default=%s", name, raw, default)
    return default


def _collect_profile_names(profiles_root: Path) -> set[str]:
    if not profiles_root.exists() or not profiles_root.is_dir():
        return set()
    return {p.name for p in profiles_root.iterdir() if p.is_dir()}


_skip_dotenv = _env_flag("REACHY_MINI_SKIP_DOTENV", default=False)
if not _skip_dotenv:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info("Configuration loaded from %s", dotenv_path)
    else:
        logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration for the dual-mode (Gemini / Local) app."""

    # ── Mode ─────────────────────────────────────────────────────────────
    # "gemini", "local" or "openai"
    APP_MODE: str = os.getenv("APP_MODE", "gemini")

    # ── Gemini ────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    # gemini-3.1-flash-live-preview is the confirmed-working Live API model on
    # v1beta (Google AI Studio).  Native-audio models (2.5-flash-native-audio-*)
    # require v1alpha and a different API key tier — use GEMINI_MODEL env var to
    # switch when that becomes available on your account.
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-live-preview")

    # ── OpenAI ────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview")

    # ── Ollama (local LLM) ────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # ── Faster-Whisper (local STT) ────────────────────────────────────────
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")

    # ── Kokoro-ONNX (local TTS) ───────────────────────────────────────────
    KOKORO_VOICE: str = os.getenv("KOKORO_VOICE", "af_heart")
    KOKORO_LANG: str = os.getenv("KOKORO_LANG", "en-us")

    # ── Vision ────────────────────────────────────────────────────────────
    HF_HOME: str = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL: str = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN: str | None = os.getenv("HF_TOKEN")

    # ── Session / VAD ────────────────────────────────────────────────────
    SESSION_DB_PATH: str | None = os.getenv("SESSION_DB_PATH")
    VAD_SILENCE_DURATION_MS: int = int(os.getenv("VAD_SILENCE_DURATION_MS", "0"))  # 0 = use backend default
    VAD_PREFIX_PADDING_MS: int = int(os.getenv("VAD_PREFIX_PADDING_MS", "0"))

    # ── English Teacher users ─────────────────────────────────────────────
    # Comma-separated list of allowed user names, e.g. "Alice,Bob".
    # The AI will identify the speaker by voice and confirm against this list.
    # Leave empty to let the AI ask for any name freely.
    _users_env = os.getenv("ENGLISH_TEACHER_USERS", "")
    ENGLISH_TEACHER_USERS: list[str] = (
        [u.strip() for u in _users_env.split(",") if u.strip()]
        if _users_env.strip()
        else []
    )

    # ── Wake / Sleep word ─────────────────────────────────────────────────
    # Phrase that wakes the robot from sleep mode (local STT, faster-whisper tiny).
    WAKE_WORD: str = os.getenv("WAKE_WORD", "hi zizi")
    # Phrase that sends the robot to sleep (detected in Gemini transcript).
    SLEEP_WORD: str = os.getenv("SLEEP_WORD", "goodbye zizi")
    # Seconds of user inactivity before automatic sleep (0 = disabled).
    SLEEP_TIMEOUT_SECONDS: int = int(os.getenv("SLEEP_TIMEOUT_SECONDS", "300"))
    # Start the app in sleep mode; robot wakes only when WAKE_WORD is heard.
    START_ASLEEP: bool = _env_flag("START_ASLEEP", default=True)

    # ── Profiles / tools ─────────────────────────────────────────────────
    _profiles_directory_env = os.getenv("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY")
    PROFILES_DIRECTORY: Path = (
        Path(_profiles_directory_env) if _profiles_directory_env else DEFAULT_PROFILES_DIRECTORY
    )
    _tools_directory_env = os.getenv("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    TOOLS_DIRECTORY: Path | None = Path(_tools_directory_env) if _tools_directory_env else None
    AUTOLOAD_EXTERNAL_TOOLS: bool = _env_flag("AUTOLOAD_EXTERNAL_TOOLS", default=False)
    REACHY_MINI_CUSTOM_PROFILE: str | None = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")


config = Config()


def set_custom_profile(profile: str | None) -> None:
    """Update the selected profile at runtime."""
    if LOCKED_PROFILE is not None:
        return
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        if profile:
            os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass

