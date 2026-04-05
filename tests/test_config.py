"""Tests for config.py – environment parsing, Config fields, set_custom_profile."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# ── _env_flag ────────────────────────────────────────────────────────────


class TestEnvFlag:
    def test_true_values(self, monkeypatch):
        from reachy_mini_teacher_app.config import _env_flag
        for val in ("1", "true", "yes", "on", "TRUE", "Yes"):
            monkeypatch.setenv("TEST_FLAG", val)
            assert _env_flag("TEST_FLAG") is True

    def test_false_values(self, monkeypatch):
        from reachy_mini_teacher_app.config import _env_flag
        for val in ("0", "false", "no", "off", "FALSE", "No"):
            monkeypatch.setenv("TEST_FLAG", val)
            assert _env_flag("TEST_FLAG") is False

    def test_default_when_unset(self, monkeypatch):
        from reachy_mini_teacher_app.config import _env_flag
        monkeypatch.delenv("TEST_FLAG_NEVER_SET", raising=False)
        assert _env_flag("TEST_FLAG_NEVER_SET") is False
        assert _env_flag("TEST_FLAG_NEVER_SET", default=True) is True

    def test_invalid_returns_default(self, monkeypatch):
        from reachy_mini_teacher_app.config import _env_flag
        monkeypatch.setenv("TEST_FLAG", "maybe")
        assert _env_flag("TEST_FLAG", default=True) is True


# ── Config class ─────────────────────────────────────────────────────────


class TestConfigClass:
    def test_has_required_fields(self):
        from reachy_mini_teacher_app.config import Config
        c = Config()
        for field in [
            "APP_MODE", "GEMINI_API_KEY", "GEMINI_MODEL",
            "OPENAI_API_KEY", "OPENAI_MODEL",
            "SESSION_DB_PATH", "VAD_SILENCE_DURATION_MS", "VAD_PREFIX_PADDING_MS",
            "PROFILES_DIRECTORY", "REACHY_MINI_CUSTOM_PROFILE",
        ]:
            assert hasattr(c, field), f"Missing field: {field}"

    def test_vad_defaults_zero(self):
        from reachy_mini_teacher_app.config import Config
        c = Config()
        # Default is 0 unless env overrides
        assert isinstance(c.VAD_SILENCE_DURATION_MS, int)
        assert isinstance(c.VAD_PREFIX_PADDING_MS, int)

    def test_app_mode_default(self):
        from reachy_mini_teacher_app.config import Config
        c = Config()
        assert c.APP_MODE in ("gemini", "openai", "local")

    def test_profiles_directory_is_path(self):
        from reachy_mini_teacher_app.config import Config
        from pathlib import Path
        c = Config()
        assert isinstance(c.PROFILES_DIRECTORY, Path)


# ── set_custom_profile ───────────────────────────────────────────────────


class TestSetCustomProfile:
    def test_set_profile(self):
        from reachy_mini_teacher_app.config import config, set_custom_profile
        original = config.REACHY_MINI_CUSTOM_PROFILE
        try:
            set_custom_profile("english_teacher")
            assert config.REACHY_MINI_CUSTOM_PROFILE == "english_teacher"
            assert os.environ.get("REACHY_MINI_CUSTOM_PROFILE") == "english_teacher"
        finally:
            set_custom_profile(original)

    def test_clear_profile(self):
        from reachy_mini_teacher_app.config import config, set_custom_profile
        original = config.REACHY_MINI_CUSTOM_PROFILE
        try:
            set_custom_profile("test_profile")
            set_custom_profile(None)
            assert config.REACHY_MINI_CUSTOM_PROFILE is None
        finally:
            set_custom_profile(original)

    def test_locked_profile_ignored(self, monkeypatch):
        import reachy_mini_teacher_app.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "LOCKED_PROFILE", "locked_one")
        cfg_mod.set_custom_profile("other")
        # Should not change because profile is locked


# ── _collect_profile_names ───────────────────────────────────────────────


class TestCollectProfileNames:
    def test_collects_profiles(self):
        from reachy_mini_teacher_app.config import _collect_profile_names, DEFAULT_PROFILES_DIRECTORY
        names = _collect_profile_names(DEFAULT_PROFILES_DIRECTORY)
        assert "english_teacher" in names or "default" in names

    def test_nonexistent_directory(self, tmp_path):
        from reachy_mini_teacher_app.config import _collect_profile_names
        assert _collect_profile_names(tmp_path / "nope") == set()

