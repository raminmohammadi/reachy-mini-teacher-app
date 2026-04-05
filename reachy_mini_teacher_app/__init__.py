"""Reachy Mini Local App – Gemini Live + Local AI Pipeline.

The Reachy Mini Apps dashboard discovers the app by importing this package
and looking for a ``ReachyMiniApp`` subclass.  Exposing ``ReachyMiniTeacherApp``
here is therefore required for HuggingFace Space publishing.
"""

from reachy_mini_teacher_app.main import ReachyMiniTeacherApp

__all__ = ["ReachyMiniTeacherApp"]

