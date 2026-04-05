"""Reachy Mini Teacher App – Gemini Live + Local AI Pipeline.

The Reachy Mini Apps dashboard discovers the app by importing this package
and looking for a ``ReachyMiniApp`` subclass.  ``ReachyMiniTeacherApp`` is
exposed here for that purpose.

Implementation note
-------------------
``main.py`` imports the robot SDK (``reachy_mini``) and several hardware-
specific modules (``moves``, ``audio``, etc.) that are only available on the
physical robot.  To keep the package importable in CI and on developer
machines without robot hardware, we use a lazy ``__getattr__`` so that
``import reachy_mini_teacher_app`` does **not** eagerly load ``main.py``.
The import is deferred until someone actually accesses ``ReachyMiniTeacherApp``
(which only happens at runtime on the robot or dashboard).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for static analysers, never executed at runtime
    from reachy_mini_teacher_app.main import ReachyMiniTeacherApp

__all__ = ["ReachyMiniTeacherApp"]


def __getattr__(name: str):  # PEP 562 — lazy module attributes
    if name == "ReachyMiniTeacherApp":
        from reachy_mini_teacher_app.main import ReachyMiniTeacherApp  # noqa: PLC0415
        return ReachyMiniTeacherApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

