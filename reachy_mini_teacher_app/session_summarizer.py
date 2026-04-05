"""Generate structured session summaries using a non-realtime LLM call.

Called at the end of a session (shutdown or profile switch) to produce a
compact summary of what was discussed/practiced and how the user performed.
The summary is stored in the DB and injected into the next session's prompt
via ``build_recap_for_user()``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from reachy_mini_teacher_app.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT = """\
You are an AI session analyst for a Farsi-speaking elderly English learner program.
You will receive a transcript and, if available, today's lesson unit details.

Produce your response in EXACTLY this format (in Farsi):

نتیجه: قبول
خلاصه: [3-6 sentence summary in Farsi]

OR if the student did not pass:

نتیجه: مردود
خلاصه: [3-6 sentence summary in Farsi]

Rules for نتیجه (pass/fail):
- Mark "قبول" ONLY if the student correctly and confidently used at least 3 of
  today's target phrases in context during the session.
- Mark "مردود" if: the session was too short, the student struggled significantly,
  fewer than 3 phrases were practiced, or the session was off-topic.

Rules for خلاصه:
- Write ONLY in Farsi.
- Cover: what phrases were practiced, how the student performed (honestly but kindly),
  and a note on what to repeat or continue next time.
- If the session was too short (< 4 messages), write:
  "جلسه بسیار کوتاه بود — موضوع خاصی تمرین نشد."
- Do NOT include greetings or filler.

{unit_section}
Transcript:
{transcript}
"""

_UNIT_SECTION_TEMPLATE = """\
Today's lesson unit: واحد {unit_id} — {unit_name}
Target phrases: {phrases}

"""

_SHORT_SESSION_RESULT = {
    "summary": "جلسه بسیار کوتاه بود — موضوع خاصی تمرین نشد.",
    "passed": False,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_session_summary(
    messages: List[Dict[str, Any]],
    daily_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a summary and pass/fail verdict from session messages.

    Args:
        messages: List of {role, content, timestamp} dicts from the session.
        daily_plan: Optional dict with keys unit_id, unit_name, phrases
                    (from SessionDB.get_or_create_daily_plan). When provided,
                    the LLM evaluates whether the student passed today's unit.

    Returns:
        Dict with keys:
            "summary" (str | None) — Farsi summary text for DB storage.
            "passed"  (bool)       — Whether the student passed today's unit.
    """
    if not messages:
        return _SHORT_SESSION_RESULT

    if len(messages) < 4:
        logger.info("Session too short (%d messages) — skipping summarization", len(messages))
        return _SHORT_SESSION_RESULT

    # Build transcript text
    lines: List[str] = []
    for msg in messages:
        role_label = "کاربر" if msg["role"] == "user" else "دستیار"
        lines.append(f"{role_label}: {msg['content']}")
    transcript = "\n".join(lines)

    # Build unit section if a daily plan was provided
    unit_section = ""
    if daily_plan:
        from reachy_mini_teacher_app.session_db import CURRICULUM
        unit_id = daily_plan.get("unit_id", 1)
        phrases = next(
            (u["phrases"] for u in CURRICULUM if u["unit_id"] == unit_id), []
        )
        unit_section = _UNIT_SECTION_TEMPLATE.format(
            unit_id=unit_id,
            unit_name=daily_plan.get("unit_name", ""),
            phrases=" · ".join(phrases),
        )

    prompt = _SUMMARIZE_PROMPT.format(
        unit_section=unit_section,
        transcript=transcript,
    )

    raw: Optional[str] = None
    if config.GEMINI_API_KEY:
        raw = await _call_gemini(prompt)
    if not raw and config.OPENAI_API_KEY:
        raw = await _call_openai(prompt)

    if not raw:
        logger.warning("No API key available or all summarization calls failed")
        return {"summary": None, "passed": False}

    return _parse_result(raw)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_result(raw: str) -> Dict[str, Any]:
    """Parse the LLM response into {summary, passed}."""
    passed = False
    summary_lines: List[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("نتیجه:"):
            verdict = stripped.replace("نتیجه:", "").strip()
            passed = "قبول" in verdict
        elif stripped.startswith("خلاصه:"):
            summary_lines.append(stripped.replace("خلاصه:", "").strip())
        elif summary_lines:
            # Continuation lines of the summary
            summary_lines.append(stripped)

    summary = " ".join(s for s in summary_lines if s) or raw.strip()
    logger.info("Session verdict: %s | summary length: %d chars", "PASS" if passed else "FAIL", len(summary))
    return {"summary": summary, "passed": passed}


async def _call_gemini(prompt: str) -> Optional[str]:
    """Call Gemini non-realtime API."""
    try:
        from google import genai

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text
        if text:
            logger.info("Session summary generated via Gemini (%d chars)", len(text))
            return text.strip()
    except Exception as e:
        logger.error("Gemini summarization failed: %s", e)
    return None


async def _call_openai(prompt: str) -> Optional[str]:
    """Call OpenAI chat completions API."""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
        )
        text = response.choices[0].message.content
        if text:
            logger.info("Session summary generated via OpenAI (%d chars)", len(text))
            return text.strip()
    except Exception as e:
        logger.error("OpenAI summarization failed: %s", e)
    return None
