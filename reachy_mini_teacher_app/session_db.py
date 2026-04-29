"""SQLite-backed session database for conversation history and per-user learning summaries.

Designed for the English-teacher persona but generic enough for any persona
that needs cross-session memory.
"""

from __future__ import annotations

import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Store the database next to the project root so it is easy to inspect,
# back up, and share between runs without hunting through the home directory.
# Path layout:  <project_root>/reachy_mini_teacher_app/session_db.py
#               → .parent       = reachy_mini_teacher_app/
#               → .parent.parent = <project_root>/
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "sessions.db"

# ---------------------------------------------------------------------------
# Curriculum definition — single source of truth for unit IDs, names, phrases
# ---------------------------------------------------------------------------

CURRICULUM: List[Dict[str, Any]] = [
    {
        "unit_id": 1,
        "unit_name": "احوالپرسی پایه",
        "phrases": ["Hello", "Hi", "Good morning", "Good evening", "How are you?",
                    "I'm fine, thank you", "Goodbye", "Bye"],
    },
    {
        "unit_id": 2,
        "unit_name": "معرفی خود",
        "phrases": ["My name is …", "I'm from Iran", "I live in …",
                    "Nice to meet you", "I'm … years old"],
    },
    {
        "unit_id": 3,
        "unit_name": "اعداد و زمان",
        "phrases": ["One, two, three … twenty", "What time is it?",
                    "It's … o'clock", "Today is Monday / Tuesday …"],
    },
    {
        "unit_id": 4,
        "unit_name": "خانواده",
        "phrases": ["This is my son", "This is my daughter",
                    "I have … children", "My husband", "My wife", "He/She is …"],
    },
    {
        "unit_id": 5,
        "unit_name": "خرید و رستوران",
        "phrases": ["How much is this?", "I want …", "Do you have …?",
                    "That's too expensive", "Thank you"],
    },
    {
        "unit_id": 6,
        "unit_name": "سلامتی و اورژانس",
        "phrases": ["I feel sick", "I have a headache", "I need a doctor",
                    "Call an ambulance", "Help me, please"],
    },
    {
        "unit_id": 7,
        "unit_name": "آب‌وهوا",
        "phrases": ["It's sunny", "It's rainy", "It's cold", "It's hot",
                    "What's the weather like?", "I like this weather"],
    },
]


class SessionDB:
    """Lightweight wrapper around a SQLite database for session memory."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._open()
        self._migrate()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _open(self) -> None:
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        logger.info("SessionDB opened at %s", self._db_path)

    def _migrate(self) -> None:
        """Create tables if they don't exist."""
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER REFERENCES users(id),
                started_at  TEXT NOT NULL DEFAULT (datetime('now')),
                ended_at    TEXT,
                summary     TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id),
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                timestamp   TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS daily_plans (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                date          TEXT NOT NULL UNIQUE,
                unit_id       INTEGER NOT NULL,
                unit_name     TEXT NOT NULL,
                passed        INTEGER NOT NULL DEFAULT 0,
                session_count INTEGER NOT NULL DEFAULT 0
            );
        """)
        self._conn.commit()
        # Additive migrations — safe to run every startup (idempotent).
        # ALTER TABLE fails silently if the column already exists.
        try:
            self._conn.execute(
                "ALTER TABLE users ADD COLUMN level INTEGER NOT NULL DEFAULT 1"
            )
            self._conn.commit()
            logger.info("Migration: added 'level' column to users table")
        except Exception:
            pass  # Column already exists — nothing to do

    def close(self) -> None:
        if self._conn:
            try:
                # Merge WAL data back into the main file so SQLite GUI tools see it.
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------

    def get_or_create_user(self, name: str) -> int:
        """Return user id, creating the row if needed."""
        assert self._conn is not None
        name_lower = name.strip().lower()
        row = self._conn.execute(
            "SELECT id FROM users WHERE LOWER(name) = ?", (name_lower,)
        ).fetchone()
        if row:
            return int(row["id"])
        cur = self._conn.execute(
            "INSERT INTO users (name) VALUES (?)", (name.strip(),)
        )
        self._conn.commit()
        user_id = cur.lastrowid
        assert user_id is not None
        logger.info("Created new user '%s' (id=%d)", name, user_id)
        return int(user_id)

    def list_users(self) -> List[Dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute("SELECT id, name, created_at, level FROM users").fetchall()
        return [dict(r) for r in rows]

    def get_user_level(self, name: str) -> int:
        """Return the student's level (1=beginner, 2=intermediate, 3=advanced).

        Defaults to 1 when the user is new or level has never been set.
        """
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT level FROM users WHERE LOWER(name) = ?", (name.strip().lower(),)
        ).fetchone()
        return max(1, min(3, int(row["level"]))) if row and row["level"] else 1

    def set_user_level(self, name: str, level: int) -> None:
        """Persist the student's assessed level (clamped to 1–3)."""
        assert self._conn is not None
        level = max(1, min(3, int(level)))
        self._conn.execute(
            "UPDATE users SET level = ? WHERE LOWER(name) = ?",
            (level, name.strip().lower()),
        )
        self._conn.commit()
        logger.info("Updated level for user '%s' → %d", name, level)

    def get_user_name_for_session(self, session_id: int) -> Optional[str]:
        """Return the display name of the user linked to a session, or None."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT u.name FROM sessions s JOIN users u ON s.user_id = u.id "
            "WHERE s.id = ?",
            (session_id,),
        ).fetchone()
        return row["name"] if row else None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self, user_id: int | None = None) -> int:
        """Create a new session row and return its id."""
        assert self._conn is not None
        cur = self._conn.execute(
            "INSERT INTO sessions (user_id) VALUES (?)", (user_id,)
        )
        self._conn.commit()
        session_id = cur.lastrowid
        assert session_id is not None
        logger.info("Session %d started (user_id=%s)", session_id, user_id)
        return int(session_id)

    def end_session(self, session_id: int, summary: str | None = None) -> None:
        assert self._conn is not None
        self._conn.execute(
            "UPDATE sessions SET ended_at = datetime('now'), summary = ? WHERE id = ?",
            (summary, session_id),
        )
        self._conn.commit()

    def assign_session_user(self, session_id: int, user_id: int) -> None:
        """Update the user_id for an existing session (called after identification)."""
        assert self._conn is not None
        self._conn.execute(
            "UPDATE sessions SET user_id = ? WHERE id = ?", (user_id, session_id)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(self, session_id: int, role: str, content: str) -> None:
        """Append a message to the current session."""
        assert self._conn is not None
        if not content or not content.strip():
            return
        self._conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content.strip()),
        )
        self._conn.commit()

    def get_session_messages(self, session_id: int) -> List[Dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Summaries & history for prompt injection
    # ------------------------------------------------------------------

    def get_latest_summary(self, user_id: int) -> Optional[str]:
        """Return the most recent non-null, non-stub summary for a user."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT summary FROM sessions "
            "WHERE user_id = ? AND summary IS NOT NULL "
            "  AND summary != 'Profile switched' "
            "  AND length(trim(summary)) > 0 "
            "ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return row["summary"] if row else None

    def get_user_session_count(self, user_id: int) -> int:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def unlink_session_user(self, session_id: int) -> None:
        """Remove the user association from a session.

        Called when the user says a switch-user trigger phrase mid-session,
        so that the next ``get_most_recent_user_name()`` query won't return
        the previous user's name.
        """
        assert self._conn is not None
        self._conn.execute(
            "UPDATE sessions SET user_id = NULL WHERE id = ?", (session_id,)
        )
        self._conn.commit()
        logger.info("Session %d unlinked from user (user switch requested)", session_id)

    def get_most_recent_user_name(self) -> Optional[str]:
        """Return the name of the user from the most recent session that has a linked user.

        Returns None if no session has ever been linked to a named user yet.
        """
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT u.name FROM sessions s "
            "JOIN users u ON s.user_id = u.id "
            "WHERE s.user_id IS NOT NULL "
            "ORDER BY s.id DESC LIMIT 1"
        ).fetchone()
        return row["name"] if row else None

    def build_recap_for_user(self, user_id: int) -> Optional[str]:
        """Build a recap string suitable for prompt injection.

        Returns None if no previous meaningful data exists.
        """
        summary = self.get_latest_summary(user_id)
        session_count = self.get_user_session_count(user_id)
        if not summary and session_count <= 1:
            return None

        parts: List[str] = []
        if session_count > 1:
            parts.append(f"این جلسه شماره {session_count} با این کاربر است.")
        if summary:
            parts.append(
                f"خلاصه جلسه قبلی:\n{summary}\n\n"
                "بر اساس این خلاصه:\n"
                "- اگر فیلد «تکرار» عبارتی دارد، آن را در ابتدای جلسه مرور کن.\n"
                "- اگر فیلد «ادامه» موضوعی دارد، بعد از مرور ادامه بده.\n"
                "- اگر «نتیجه» مردود بود، همان واحد را دوباره از ابتدا شروع کن."
            )
        return "\n".join(parts)

    def get_recent_session_recap(
        self,
        current_session_id: int | None = None,
        max_past: int = 2,
        user_id: int | None = None,
    ) -> Optional[str]:
        """Return a combined recap of the last N completed sessions for a specific user.

        Filters by user_id when provided so users never see each other's summaries.
        Skips stub summaries and the currently-open session.
        """
        assert self._conn is not None
        _STUB_SUMMARIES = frozenset([
            "Profile switched",
            "جلسه بسیار کوتاه بود — موضوع خاصی تمرین نشد.",
        ])
        conditions = [
            "summary IS NOT NULL",
            "length(trim(summary)) > 20",
        ]
        params: list = []

        # Filter by user when we know who this is
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)

        if current_session_id is not None:
            conditions.append("id != ?")
            params.append(current_session_id)
        params.append(max_past)

        where_clause = " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT summary FROM sessions WHERE {where_clause} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()

        # Filter out stub summaries in Python (avoids complex SQL LIKE chains)
        real_summaries = [
            r["summary"] for r in reversed(rows)
            if r["summary"] not in _STUB_SUMMARIES
            and not r["summary"].startswith("جلسه بسیار کوتاه")
        ]

        if not real_summaries:
            return None

        if len(real_summaries) == 1:
            return f"خلاصه جلسه قبلی:\n{real_summaries[0]}"

        parts = [f"جلسه {i + 1} (اخیر):\n{s}" for i, s in enumerate(real_summaries)]
        return "خلاصه جلسات اخیر (از قدیم به جدید):\n\n" + "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Daily plan — curriculum scheduling
    # ------------------------------------------------------------------

    def get_or_create_daily_plan(self, date: str) -> Dict[str, Any]:
        """Return today's unit plan, creating it if it doesn't exist.

        The unit is chosen by checking the most recent previous day:
        - If that day was passed → advance to the next unit.
        - If not passed (or no history) → repeat the same unit.
        - Very first day ever → start at unit 1.
        """
        assert self._conn is not None

        row = self._conn.execute(
            "SELECT * FROM daily_plans WHERE date = ?", (date,)
        ).fetchone()
        if row:
            return dict(row)

        # Determine unit for today based on the last completed day
        unit_id = 1
        last = self._conn.execute(
            "SELECT unit_id, passed FROM daily_plans WHERE date < ? ORDER BY date DESC LIMIT 1",
            (date,),
        ).fetchone()
        if last:
            if last["passed"]:
                unit_id = (last["unit_id"] % len(CURRICULUM)) + 1
            else:
                unit_id = last["unit_id"]

        unit_name = next(
            (u["unit_name"] for u in CURRICULUM if u["unit_id"] == unit_id), "احوالپرسی پایه"
        )
        self._conn.execute(
            "INSERT INTO daily_plans (date, unit_id, unit_name) VALUES (?, ?, ?)",
            (date, unit_id, unit_name),
        )
        self._conn.commit()
        logger.info("Daily plan created for %s → unit %d (%s)", date, unit_id, unit_name)
        return dict(self._conn.execute(
            "SELECT * FROM daily_plans WHERE date = ?", (date,)
        ).fetchone())

    def increment_daily_session_count(self, date: str) -> None:
        """Increment the number of sessions the student had on this date."""
        assert self._conn is not None
        self._conn.execute(
            "UPDATE daily_plans SET session_count = session_count + 1 WHERE date = ?",
            (date,),
        )
        self._conn.commit()

    def mark_daily_plan_result(self, date: str, passed: bool) -> None:
        """Record whether the student passed today's unit.

        A day is only marked 'passed' if it hasn't been passed before,
        so multiple sessions on the same day cannot downgrade a pass.
        """
        assert self._conn is not None
        if passed:
            self._conn.execute(
                "UPDATE daily_plans SET passed = 1 WHERE date = ?", (date,)
            )
        else:
            # Only update to 0 if it hasn't already been passed today
            self._conn.execute(
                "UPDATE daily_plans SET passed = 0 WHERE date = ? AND passed = 0", (date,)
            )
        self._conn.commit()
        logger.info("Daily plan %s → %s", date, "PASSED ✓" if passed else "not passed yet")

    def get_daily_plan_for_prompt(self, date: str) -> str:
        """Return a formatted string for injection into the system prompt."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT unit_id, unit_name, session_count, passed FROM daily_plans WHERE date = ?",
            (date,),
        ).fetchone()
        if not row:
            return ""

        unit_id   = row["unit_id"]
        unit_name = row["unit_name"]
        session_count = row["session_count"]

        phrases = next(
            (u["phrases"] for u in CURRICULUM if u["unit_id"] == unit_id), []
        )
        phrases_str = " · ".join(phrases)

        repeat_note = ""
        if session_count >= 1:
            repeat_note = (
                f"\nکاربر امروز قبلاً {session_count} جلسه روی این موضوع کار کرده. "
                "همان عبارات را دوباره مرور کن — ممکن است بعضی را فراموش کرده باشد."
            )

        return (
            f"## برنامه درس امروز\n"
            f"**واحد {unit_id} — {unit_name}**\n"
            f"عبارات هدف: {phrases_str}\n\n"
            f"🔒 فقط روی این واحد تمرکز کن. امروز به واحد دیگری نرو.{repeat_note}"
        )

    def build_current_session_transcript(
        self, session_id: int, max_messages: int = 50
    ) -> Optional[str]:
        """Build a transcript of the current session's messages for prompt injection.

        Returns None if no messages exist yet for this session.
        The transcript is formatted as a simple dialogue so the AI can
        continue the conversation seamlessly after a reconnection.
        """
        messages = self.get_session_messages(session_id)
        if not messages:
            return None

        # Take only the last N messages to avoid exceeding token limits
        recent = messages[-max_messages:]
        lines: List[str] = []
        for msg in recent:
            role_label = "کاربر" if msg["role"] == "user" else "دستیار"
            lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(lines)

