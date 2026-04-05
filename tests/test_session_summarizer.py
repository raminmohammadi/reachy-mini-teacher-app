"""Tests for session_summarizer — pass/fail parsing and summary building."""
from __future__ import annotations

import pytest
from reachy_mini_teacher_app.session_summarizer import (
    _parse_result,
    _SHORT_SESSION_RESULT,
    generate_session_summary,
)


# ── _parse_result ────────────────────────────────────────────────────────────

class TestParseResult:
    def test_pass_verdict(self):
        raw = "نتیجه: قبول\nخلاصه: کاربر خوب یاد گرفت."
        r = _parse_result(raw)
        assert r["passed"] is True
        assert "خوب یاد گرفت" in r["summary"]

    def test_fail_verdict(self):
        raw = "نتیجه: مردود\nخلاصه: جلسه کوتاه بود."
        r = _parse_result(raw)
        assert r["passed"] is False
        assert "جلسه کوتاه" in r["summary"]

    def test_pass_with_extra_whitespace(self):
        raw = "نتیجه:  قبول  \nخلاصه: آفرین!"
        assert _parse_result(raw)["passed"] is True

    def test_fail_with_extra_whitespace(self):
        raw = "نتیجه:  مردود  \nخلاصه: ضعیف بود."
        assert _parse_result(raw)["passed"] is False

    def test_multiline_summary_joined(self):
        raw = "نتیجه: قبول\nخلاصه: خط اول.\nخط دوم."
        r = _parse_result(raw)
        assert "خط اول" in r["summary"]
        assert "خط دوم" in r["summary"]

    def test_malformed_falls_back_to_raw(self):
        raw = "این متن هیچ فرمتی ندارد"
        r = _parse_result(raw)
        assert r["passed"] is False
        assert raw.strip() in r["summary"]

    def test_empty_string_falls_back(self):
        r = _parse_result("")
        assert r["passed"] is False
        assert isinstance(r["summary"], str)

    def test_pass_with_قبول_anywhere_in_verdict(self):
        raw = "نتیجه: بله، قبول شد\nخلاصه: عالی بود."
        assert _parse_result(raw)["passed"] is True


# ── Short-session constant ───────────────────────────────────────────────────

class TestShortSessionResult:
    def test_has_required_keys(self):
        assert "summary" in _SHORT_SESSION_RESULT
        assert "passed" in _SHORT_SESSION_RESULT

    def test_passed_is_false(self):
        assert _SHORT_SESSION_RESULT["passed"] is False

    def test_summary_is_string(self):
        assert isinstance(_SHORT_SESSION_RESULT["summary"], str)


# ── generate_session_summary (no network calls) ──────────────────────────────

class TestGenerateSessionSummary:
    """Test the function's early-exit logic without making real API calls."""

    @pytest.mark.asyncio
    async def test_empty_messages_returns_short_result(self):
        result = await generate_session_summary([])
        assert result["passed"] is False
        assert result["summary"] == _SHORT_SESSION_RESULT["summary"]

    @pytest.mark.asyncio
    async def test_too_few_messages_returns_short_result(self):
        msgs = [{"role": "user", "content": "سلام", "timestamp": ""}] * 3
        result = await generate_session_summary(msgs)
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_returns_dict_with_required_keys(self, monkeypatch):
        """With no API keys configured, should return {summary: None, passed: False}."""
        monkeypatch.setattr("reachy_mini_teacher_app.session_summarizer.config.GEMINI_API_KEY", None)
        monkeypatch.setattr("reachy_mini_teacher_app.session_summarizer.config.OPENAI_API_KEY", None)
        msgs = [{"role": "user", "content": f"msg {i}", "timestamp": ""} for i in range(6)]
        result = await generate_session_summary(msgs)
        assert "summary" in result
        assert "passed" in result
        assert result["passed"] is False  # no API → can't evaluate

    @pytest.mark.asyncio
    async def test_daily_plan_included_in_prompt_build(self, monkeypatch):
        """Ensure that daily_plan is passed through without raising errors."""
        monkeypatch.setattr("reachy_mini_teacher_app.session_summarizer.config.GEMINI_API_KEY", None)
        monkeypatch.setattr("reachy_mini_teacher_app.session_summarizer.config.OPENAI_API_KEY", None)
        msgs = [{"role": "user", "content": f"msg {i}", "timestamp": ""} for i in range(6)]
        plan = {"unit_id": 1, "unit_name": "احوالپرسی پایه"}
        result = await generate_session_summary(msgs, daily_plan=plan)
        assert "passed" in result
