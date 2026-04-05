"""Tests for prompts.py – instruction loading, placeholder expansion, voice resolution."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from reachy_mini_teacher_app.prompts import (
    _expand_prompt_includes,
    get_session_instructions,
    get_session_voice,
    PROMPTS_LIBRARY_DIRECTORY,
)


# ── _expand_prompt_includes ──────────────────────────────────────────────


class TestExpandPromptIncludes:
    def test_no_placeholders(self):
        text = "Hello world\nNo placeholders here"
        assert _expand_prompt_includes(text) == text

    def test_placeholder_expanded(self, tmp_path, monkeypatch):
        # Create a fake prompts library with a template
        lib = tmp_path / "prompts"
        lib.mkdir()
        (lib / "greeting.txt").write_text("Welcome to the robot!", encoding="utf-8")
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.PROMPTS_LIBRARY_DIRECTORY", lib)
        result = _expand_prompt_includes("[greeting]")
        assert "Welcome to the robot!" in result

    def test_missing_template_kept(self, tmp_path, monkeypatch):
        lib = tmp_path / "prompts"
        lib.mkdir()
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.PROMPTS_LIBRARY_DIRECTORY", lib)
        result = _expand_prompt_includes("[nonexistent]")
        assert "[nonexistent]" in result

    def test_non_placeholder_brackets_ignored(self):
        text = "Use [this] inline"
        # "[this]" on a line with other text should not be treated as a placeholder
        assert _expand_prompt_includes(text) == text

    def test_subdirectory_placeholder(self, tmp_path, monkeypatch):
        lib = tmp_path / "prompts"
        (lib / "sub").mkdir(parents=True)
        (lib / "sub" / "deep.txt").write_text("Deep content", encoding="utf-8")
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.PROMPTS_LIBRARY_DIRECTORY", lib)
        result = _expand_prompt_includes("[sub/deep]")
        assert "Deep content" in result


# ── get_session_instructions ─────────────────────────────────────────────


class TestGetSessionInstructions:
    def test_default_profile_loads(self, monkeypatch):
        """When no custom profile is set, the default prompt should load."""
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", None)
        # The default prompt file must exist in the source tree
        instructions = get_session_instructions()
        assert len(instructions) > 0

    def test_english_teacher_profile(self, monkeypatch):
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", "english_teacher")
        monkeypatch.setattr(
            "reachy_mini_teacher_app.prompts.config.PROFILES_DIRECTORY",
            Path(__file__).resolve().parent.parent / "reachy_mini_teacher_app" / "profiles",
        )
        instructions = get_session_instructions()
        assert "فارسی" in instructions or "ریچی" in instructions

    def test_recap_injected(self, monkeypatch, tmp_path):
        # Use a synthetic profile that has the {previous_session_recap} placeholder.
        # (english_teacher was redesigned to use {todays_plan} instead.)
        profile_dir = tmp_path / "recap_profile"
        profile_dir.mkdir()
        (profile_dir / "instructions.txt").write_text(
            "درس: {previous_session_recap}", encoding="utf-8"
        )
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", "recap_profile")
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.PROFILES_DIRECTORY", tmp_path)
        recap = "کاربر Hello و Goodbye یاد گرفت"
        instructions = get_session_instructions(recap=recap)
        assert "Hello و Goodbye" in instructions

    def test_no_recap_default_text(self, monkeypatch, tmp_path):
        profile_dir = tmp_path / "recap_profile"
        profile_dir.mkdir()
        (profile_dir / "instructions.txt").write_text(
            "درس: {previous_session_recap}", encoding="utf-8"
        )
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", "recap_profile")
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.PROFILES_DIRECTORY", tmp_path)
        instructions = get_session_instructions(recap=None)
        assert "اولین جلسه" in instructions

    def test_missing_profile_exits(self, monkeypatch):
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", "nonexistent_profile_xyz")
        monkeypatch.setattr(
            "reachy_mini_teacher_app.prompts.config.PROFILES_DIRECTORY",
            Path(__file__).resolve().parent.parent / "reachy_mini_teacher_app" / "profiles",
        )
        with pytest.raises(SystemExit):
            get_session_instructions()

    def test_user_name_injected(self, monkeypatch):
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", "english_teacher")
        monkeypatch.setattr(
            "reachy_mini_teacher_app.prompts.config.PROFILES_DIRECTORY",
            Path(__file__).resolve().parent.parent / "reachy_mini_teacher_app" / "profiles",
        )
        instructions = get_session_instructions(user_name="علی")
        assert "علی" in instructions
        assert "{known_user_name}" not in instructions

    def test_no_user_name_placeholder_cleared(self, monkeypatch):
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", "english_teacher")
        monkeypatch.setattr(
            "reachy_mini_teacher_app.prompts.config.PROFILES_DIRECTORY",
            Path(__file__).resolve().parent.parent / "reachy_mini_teacher_app" / "profiles",
        )
        instructions = get_session_instructions(user_name=None)
        assert "{known_user_name}" not in instructions

    def test_daily_plan_injected(self, monkeypatch):
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", "english_teacher")
        monkeypatch.setattr(
            "reachy_mini_teacher_app.prompts.config.PROFILES_DIRECTORY",
            Path(__file__).resolve().parent.parent / "reachy_mini_teacher_app" / "profiles",
        )
        plan = "## برنامه درس امروز\nواحد 1 — احوالپرسی پایه"
        instructions = get_session_instructions(daily_plan=plan)
        assert "واحد 1" in instructions
        assert "{todays_plan}" not in instructions

    def test_no_daily_plan_placeholder_cleared(self, monkeypatch):
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", "english_teacher")
        monkeypatch.setattr(
            "reachy_mini_teacher_app.prompts.config.PROFILES_DIRECTORY",
            Path(__file__).resolve().parent.parent / "reachy_mini_teacher_app" / "profiles",
        )
        instructions = get_session_instructions(daily_plan=None)
        assert "{todays_plan}" not in instructions


# ── get_session_voice ────────────────────────────────────────────────────


class TestGetSessionVoice:
    def test_default_voice(self, monkeypatch):
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", None)
        assert get_session_voice() == "cedar"

    def test_custom_default(self, monkeypatch):
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", None)
        assert get_session_voice(default="puck") == "puck"

    def test_english_teacher_voice(self, monkeypatch):
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", "english_teacher")
        monkeypatch.setattr(
            "reachy_mini_teacher_app.prompts.config.PROFILES_DIRECTORY",
            Path(__file__).resolve().parent.parent / "reachy_mini_teacher_app" / "profiles",
        )
        voice = get_session_voice()
        assert voice  # should be non-empty (e.g. "Aoede")

    def test_missing_voice_returns_default(self, monkeypatch, tmp_path):
        (tmp_path / "no_voice_profile").mkdir()
        (tmp_path / "no_voice_profile" / "instructions.txt").write_text("hi", encoding="utf-8")
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.REACHY_MINI_CUSTOM_PROFILE", "no_voice_profile")
        monkeypatch.setattr("reachy_mini_teacher_app.prompts.config.PROFILES_DIRECTORY", tmp_path)
        assert get_session_voice() == "cedar"

