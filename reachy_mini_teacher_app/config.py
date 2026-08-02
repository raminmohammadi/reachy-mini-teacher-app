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
    # Comma-separated list of allowed users. Each entry supports:
    #   Name                       — canonical name, no extras
    #   Name:m   /  Name:f         — canonical name + gender hint
    #   Name/alias1/alias2:m       — canonical name + gender + one or more
    #                                aliases (e.g. Persian-script spelling)
    # Aliases collapse into the canonical name when the user is identified,
    # so mis-hearings and script variants don't create duplicate DB rows.
    # Examples:
    #     ENGLISH_TEACHER_USERS=Alice,Bob
    #     ENGLISH_TEACHER_USERS=Bob:m,Alice:f
    #     ENGLISH_TEACHER_USERS=Bob/باب/Bab:m,Alice/آلیس:f
    _users_env = os.getenv("ENGLISH_TEACHER_USERS", "")

    @staticmethod
    def _parse_users_env(raw: str) -> list[dict]:
        """Parse ``Name[/alias…][:gender]`` entries.

        Returns ``[{"name": ..., "gender": ..., "aliases": [...]}, …]``.
        """
        entries: list[dict] = []
        if not raw or not raw.strip():
            return entries
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                head, _, gender = token.partition(":")
                gender = gender.strip().lower()
                if gender not in {"m", "f"}:
                    gender = ""
            else:
                head, gender = token, ""
            parts = [p.strip() for p in head.split("/") if p.strip()]
            if not parts:
                continue
            name, aliases = parts[0], parts[1:]
            entries.append({"name": name, "gender": gender, "aliases": aliases})
        return entries

    ENGLISH_TEACHER_USER_ENTRIES: list[dict] = _parse_users_env.__func__(_users_env)  # type: ignore[attr-defined]
    ENGLISH_TEACHER_USER_NAMES: list[str] = [e["name"] for e in ENGLISH_TEACHER_USER_ENTRIES]
    # Flat list of "heard-form → canonical" aliases used by SessionDB to
    # snap variants into the canonical row. Includes both explicit aliases
    # and each canonical name itself.
    ENGLISH_TEACHER_USER_ALIASES: dict[str, str] = {
        alias.strip().lower(): e["name"]
        for e in ENGLISH_TEACHER_USER_ENTRIES
        for alias in ([e["name"], *e.get("aliases", [])])
    }
    # Kept for backward compatibility with any external code / tests that
    # read the plain string list.
    ENGLISH_TEACHER_USERS: list[str] = list(ENGLISH_TEACHER_USER_NAMES)

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

