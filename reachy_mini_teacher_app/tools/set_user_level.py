"""Tool that lets the AI silently record the student's assessed English level.

The AI calls this once early in each session (after 2-3 exchanges) and again
any time the student's demonstrated ability clearly exceeds or falls below the
current level.  The level is persisted per user so the next session starts at
the right difficulty without re-assessing from scratch.

Levels
------
1 — مبتدی   (Beginner)    : needs full hints, translations always, slow pace
2 — متوسط   (Intermediate): occasional hints, some translation, normal pace
3 — پیشرفته  (Advanced)   : direct questions, no translation, fast pace
"""

import logging
from typing import Any, Dict

from reachy_mini_teacher_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

_LEVEL_LABELS = {1: "مبتدی", 2: "متوسط", 3: "پیشرفته"}


class SetUserLevel(Tool):
    """Silently persist the student's assessed English level (1–3)."""

    name = "set_user_level"
    description = (
        "Call this tool silently — without telling the user — after assessing "
        "their English level from the first 2-3 exchanges. "
        "Also call it again if the student clearly improves or struggles. "
        "Level 1 = beginner (مبتدی), Level 2 = intermediate (متوسط), "
        "Level 3 = advanced (پیشرفته). "
        "Do NOT announce the level to the user."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "description": (
                    "Assessed level: '1' (beginner/مبتدی), "
                    "'2' (intermediate/متوسط), or '3' (advanced/پیشرفته)."
                ),
                "enum": ["1", "2", "3"],
            },
        },
        "required": ["level"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        try:
            level = int(kwargs.get("level", 1))
        except (TypeError, ValueError):
            return {"status": "error", "reason": "level must be 1, 2, or 3"}

        db = deps.session_db
        session_state = deps.session_state
        session_id = session_state.get("session_id") if session_state else None

        if db is None:
            logger.warning("set_user_level: session_db not available in deps")
            return {"status": "ignored", "reason": "no db"}

        if session_id is None:
            logger.warning("set_user_level: session_id not set yet")
            return {"status": "ignored", "reason": "no session_id"}

        try:
            user_name = db.get_user_name_for_session(session_id)
            if not user_name:
                logger.warning("set_user_level: no user linked to session %d", session_id)
                return {"status": "ignored", "reason": "no user linked to session"}

            db.set_user_level(user_name, level)
            label = _LEVEL_LABELS.get(level, str(level))
            logger.info(
                "Level for '%s' set to %d (%s) via set_user_level tool",
                user_name, level, label,
            )
            return {"status": "ok", "level": level, "label": label}

        except Exception as exc:
            logger.error("set_user_level failed: %s", exc)
            return {"status": "error", "reason": str(exc)}
