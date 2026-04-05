"""Tool that lets the AI silently record the user's name in the session database.

The AI calls this exactly once per session — as soon as it learns the user's
name — so that future sessions can greet the user by name without asking again.
"""

import logging
from typing import Any, Dict

from reachy_mini_teacher_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class RememberUserName(Tool):
    """Silently save the user's name so future sessions remember it."""

    name = "remember_user_name"
    description = (
        "Call this tool ONCE as soon as you learn the user's name. "
        "It saves their name so you can greet them by name in future sessions. "
        "Do NOT tell the user you are calling this tool — just call it silently."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The user's first name exactly as they said it.",
            },
        },
        "required": ["name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        user_name = (kwargs.get("name") or "").strip()
        if not user_name:
            logger.warning("remember_user_name called with empty name — ignoring")
            return {"status": "ignored", "reason": "empty name"}

        db = deps.session_db
        session_state = deps.session_state
        session_id = session_state.get("session_id") if session_state else None

        if db is None:
            logger.warning("remember_user_name: session_db not available in deps")
            return {"status": "ignored", "reason": "no db"}

        try:
            user_id = db.get_or_create_user(user_name)
            if session_id is not None:
                db.assign_session_user(session_id, user_id)
                logger.info(
                    "Linked session %d to user '%s' (user_id=%d)",
                    session_id, user_name, user_id,
                )
            else:
                logger.warning("remember_user_name: session_id not set yet — user created but not linked")
        except Exception as e:
            logger.error("remember_user_name failed: %s", e)
            return {"status": "error", "reason": str(e)}

        return {"status": "ok"}
