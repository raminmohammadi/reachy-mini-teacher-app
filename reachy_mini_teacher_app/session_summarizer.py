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
You will receive a session transcript and, if available, today's lesson unit details.

Produce your response in EXACTLY this structured format (all fields in Farsi):

کاربر: [نام کاربر را از مکالمه استخراج کن. اگر مشخص نیست بنویس: «نامشخص»]
نتیجه: [قبول یا مردود]
تمرین‌شده: [فقط عبارات هدفِ **همین واحد** که کاربر تلاش کرد بگوید — از لیست «عبارات هدف» زیر انتخاب کن. عبارات خارج از این لیست را در این فیلد نیاور.]
خارج از واحد: [اگر معلم عباراتی خارج از لیست هدف را وارد جلسه کرده، آنها را اینجا فهرست کن — این نشانه‌ی پرت‌شدن از برنامه است.]
عملکرد: [ارزیابی صادقانه: کدام عبارات هدف را درست/اشتباه گفت، در چند سناریوی متفاوت تمرین شد]
تکرار: [عبارات هدفی که باید دفعه بعد دوباره تمرین شوند — با دلیل کوتاه]
ادامه: [آیا این واحد کامل شده؟ اگر بله، دفعه بعد به واحد بعدی برو؛ اگر نه، همین واحد ادامه یابد.]
یادداشت: [نکات مهم برای معلم — سرعت یادگیری، اعتماد به نفس، پرت شدن معلم از موضوع، خستگی]

قوانین نتیجه (قبول/مردود) — **فقط بر اساس عبارات هدف همین واحد**، نه هر جمله‌ی انگلیسی که در مکالمه ظاهر شده:
- «قبول»: کاربر حداقل ۳ عبارت از لیست «عبارات هدف» را در حداقل ۲ سناریوی متفاوت درست و با اطمینان استفاده کرده.
- «مردود»: کاربر کمتر از ۳ عبارت هدف را درست گفت، یا هر عبارت فقط در یک سناریو تمرین شد، یا معلم بیشتر وقت جلسه را روی موضوعات خارج از واحد گذراند.

اگر جلسه کمتر از ۴ پیام داشت، تمام فیلدها را با «جلسه بسیار کوتاه بود» پر کن و نتیجه را «مردود» بگذار.

{unit_section}
متن مکالمه:
{transcript}
"""

_UNIT_SECTION_TEMPLATE = """\
واحد درسی امروز: واحد {unit_id} — {unit_name}
عبارات هدف: {phrases}

"""

_SHORT_SESSION_RESULT = {
    "summary": (
        "کاربر: نامشخص\n"
        "نتیجه: مردود\n"
        "تمرین‌شده: —\n"
        "عملکرد: جلسه بسیار کوتاه بود — موضوع خاصی تمرین نشد.\n"
        "تکرار: همه عبارات واحد فعلی\n"
        "ادامه: از ابتدای واحد شروع کن\n"
        "یادداشت: جلسه کوتاه بود"
    ),
    "passed": False,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_session_summary(
    messages: List[Dict[str, Any]],
    daily_plan: Optional[Dict[str, Any]] = None,
    user_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a structured summary and pass/fail verdict from session messages.

    Args:
        messages:    List of {role, content, timestamp} dicts from the session.
        daily_plan:  Optional dict with keys unit_id, unit_name, phrases.
        user_name:   Known user name to embed in the summary.

    Returns:
        Dict with keys:
            "summary" (str | None) — Structured Farsi summary for DB storage.
            "passed"  (bool)       — Whether the student passed today's unit.
    """
    if not messages:
        return _SHORT_SESSION_RESULT

    if len(messages) < 4:
        logger.info("Session too short (%d messages) — skipping summarization", len(messages))
        short = dict(_SHORT_SESSION_RESULT)
        if user_name:
            short["summary"] = short["summary"].replace("کاربر: نامشخص", f"کاربر: {user_name}")
        return short

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
    """Parse the structured LLM response into {summary, passed}.

    The new format has labelled fields: کاربر, نتیجه, تمرین‌شده, عملکرد, تکرار, ادامه, یادداشت.
    We keep the full structured text as the summary (for rich prompt injection)
    and extract only the نتیجه field for the pass/fail boolean.
    """
    passed = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("نتیجه:"):
            verdict = stripped.replace("نتیجه:", "").strip()
            passed = "قبول" in verdict
            break

    summary = raw.strip()
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
