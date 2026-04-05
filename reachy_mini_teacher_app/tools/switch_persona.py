"""Tool: switch_persona — lets the AI switch the active persona/profile at runtime.

When the user says something like "switch to English teacher" or "بریم انگلیسی یاد بگیریم",
the AI calls this tool. It updates the global profile config and signals the handler
to reconnect with the new persona's instructions and tools.
"""

import logging
from typing import Any, Dict

from reachy_mini_teacher_app.config import (
    config,
    set_custom_profile,
    _collect_profile_names,
)
from reachy_mini_teacher_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

# Map of user-friendly aliases → profile directory names
_PROFILE_ALIASES: Dict[str, str] = {
    "english teacher": "english_teacher",
    "english_teacher": "english_teacher",
    "معلم انگلیسی": "english_teacher",
    "انگلیسی": "english_teacher",
    "default": "default",
    "normal": "default",
    "عادی": "default",
    "پیش‌فرض": "default",
}


class SwitchPersona(Tool):
    """Switch the robot's active persona/profile.

    Call this when the user asks to change personality, for example
    "switch to English teacher" or "go back to normal mode".
    After switching, the session will restart with the new persona's
    instructions and voice.
    """

    name = "switch_persona"
    description = (
        "Switch the robot's active persona/profile. Use when the user asks to "
        "change to a different mode like 'English teacher' or 'default'. "
        "This will restart the session with the new persona."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "persona_name": {
                "type": "string",
                "description": (
                    "Name of the persona to switch to. "
                    "Valid values: 'english_teacher', 'default'."
                ),
            },
        },
        "required": ["persona_name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        raw_name = kwargs.get("persona_name", "").strip().lower()
        logger.info("switch_persona called with persona_name=%r", raw_name)

        # Resolve alias
        profile_name = _PROFILE_ALIASES.get(raw_name, raw_name)

        # Validate profile exists
        available = _collect_profile_names(config.PROFILES_DIRECTORY)
        # "default" means clearing the custom profile (uses built-in default prompt)
        if profile_name == "default":
            profile_name = None  # type: ignore[assignment]
        elif profile_name not in available:
            return {
                "error": f"Unknown persona '{raw_name}'. Available: {sorted(available)}",
            }

        # Update config
        set_custom_profile(profile_name)
        logger.info("Profile switched to: %s", profile_name or "default")

        # Signal handler to reconnect
        event = deps.profile_switch_event
        if event is not None:
            event.set()
            logger.info("Profile switch event signalled — handler will reconnect")
        else:
            logger.warning("No profile_switch_event on deps — handler may not reconnect")

        return {
            "status": "switched",
            "new_persona": profile_name or "default",
            "message": "Session will restart with the new persona now.",
        }

