"""Tool that resets the user context so a new person can be identified.

The AI calls this silently when it detects a trigger phrase such as
«من کسی دیگری هستم» or «کاربر عوض شد».  After calling it the AI asks
for the new user's first name and then calls remember_user_name.
"""

import logging
from typing import Any, Dict

from reachy_mini_teacher_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class SwitchUser(Tool):
    """Forget the current user and prepare to learn the next one's name."""

    name = "switch_user"
    description = (
        "Call this tool silently when the user says something like "
        "«من کسی دیگری هستم», «کاربر عوض شد», «اسمم را عوض کن», "
        "«یه نفر دیگه‌ام», or any phrase indicating a different person "
        "is now using the device. "
        "After calling it, ask for the new user's first name and then "
        "call remember_user_name with their name. "
        "Do NOT tell the user you are calling this tool."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        db = deps.session_db
        session_state = deps.session_state
        session_id = session_state.get("session_id") if session_state else None

        if db is None:
            logger.warning("switch_user: session_db not available in deps")
            return {"status": "ignored", "reason": "no db"}

        if session_id is not None:
            db.unlink_session_user(session_id)

        logger.info("User switch requested — current session unlinked from previous user")
        return {"status": "ok"}
