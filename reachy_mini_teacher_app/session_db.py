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
    # ── Level 1 — Beginner (units 1-7) ────────────────────────────────────
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
    # ── Level 2 — Intermediate (units 8-14) ───────────────────────────────
    {
        "unit_id": 8,
        "unit_name": "روزمره — گذشته و آینده",
        "phrases": ["Yesterday I went to …", "Tomorrow I will …",
                    "Last week we visited …", "Next month I'm planning to …",
                    "I already did that"],
    },
    {
        "unit_id": 9,
        "unit_name": "سفر و حمل و نقل",
        "phrases": ["Where is the bus stop?", "How much is the ticket?",
                    "Does this train go to …?", "I'd like a taxi to …",
                    "What time does it leave?", "Is it far from here?"],
    },
    {
        "unit_id": 10,
        "unit_name": "تلفن و پیام",
        "phrases": ["Can I speak to …?", "Please call me back",
                    "Sorry, wrong number", "Could you repeat that?",
                    "I'll send you a message"],
    },
    {
        "unit_id": 11,
        "unit_name": "رستوران — جزئیات",
        "phrases": ["A table for two, please", "Could I see the menu?",
                    "I'll have the …", "No sugar, please",
                    "The bill, please", "It was delicious"],
    },
    {
        "unit_id": 12,
        "unit_name": "لباس و خرید",
        "phrases": ["Can I try this on?", "Do you have a bigger size?",
                    "It doesn't fit", "Is there a discount?",
                    "I'll take it"],
    },
    {
        "unit_id": 13,
        "unit_name": "جهت‌یابی",
        "phrases": ["Excuse me, where is …?", "Turn left / turn right",
                    "Go straight ahead", "It's next to …",
                    "How long does it take?"],
    },
    {
        "unit_id": 14,
        "unit_name": "علاقه و سرگرمی",
        "phrases": ["I like reading", "I enjoy walking in the park",
                    "My favorite food is …", "Do you like …?",
                    "In my free time I …"],
    },
    # ── Level 3 — Advanced (units 15-21) ──────────────────────────────────
    {
        "unit_id": 15,
        "unit_name": "بیان عقیده",
        "phrases": ["In my opinion, …", "I think that …",
                    "I agree with you, but …", "I'm not sure about that",
                    "That's a good point"],
    },
    {
        "unit_id": 16,
        "unit_name": "شکایت و توضیح مشکل",
        "phrases": ["There's a problem with …", "This isn't what I ordered",
                    "Could you help me with this?", "It's not working properly",
                    "I'd like to speak to the manager"],
    },
    {
        "unit_id": 17,
        "unit_name": "روایت گذشته با جزئیات",
        "phrases": ["When I was young, I used to …",
                    "Many years ago, I …", "That reminds me of …",
                    "I'll never forget the day …", "Back then, we …"],
    },
    {
        "unit_id": 18,
        "unit_name": "برنامه‌ریزی آینده",
        "phrases": ["I'm thinking of …", "If everything goes well, I'll …",
                    "By next year, I hope to …", "It depends on …",
                    "Let's arrange a time"],
    },
    {
        "unit_id": 19,
        "unit_name": "فرهنگ و اجتماع",
        "phrases": ["In Iran, we usually …", "Here in this country, people …",
                    "It's a tradition to …", "That's interesting — tell me more",
                    "Cultures are different, but …"],
    },
    {
        "unit_id": 20,
        "unit_name": "اصطلاحات رایج",
        "phrases": ["It's a piece of cake", "Break a leg!",
                    "Once in a blue moon", "Under the weather",
                    "Better late than never"],
    },
    {
        "unit_id": 21,
        "unit_name": "موقعیت پیچیده — پزشک و بانک",
        "phrases": ["I've been feeling … for a few days",
                    "Could you write down the dosage?",
                    "I'd like to open an account", "Where do I sign?",
                    "Is there a fee for this?"],
    },
]

# Level → unit-id range (inclusive). Level 1 is beginner, 3 is advanced.
LEVEL_UNIT_RANGES: Dict[int, tuple[int, int]] = {
    1: (1, 7),
    2: (8, 14),
    3: (15, 21),
}


def get_units_for_level(level: int) -> List[Dict[str, Any]]:
    """Return the curriculum units that belong to a given level band."""
    lo, hi = LEVEL_UNIT_RANGES.get(int(level), LEVEL_UNIT_RANGES[1])
    return [u for u in CURRICULUM if lo <= u["unit_id"] <= hi]


def _clamp_unit_to_level(unit_id: int, level: int) -> int:
    """Clamp/wrap `unit_id` so it belongs to `level`'s band."""
    lo, hi = LEVEL_UNIT_RANGES.get(int(level), LEVEL_UNIT_RANGES[1])
    if unit_id < lo or unit_id > hi:
        return lo
    return unit_id


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
                date          TEXT NOT NULL,
                user_id       INTEGER REFERENCES users(id),
                unit_id       INTEGER NOT NULL,
                unit_name     TEXT NOT NULL,
                passed        INTEGER NOT NULL DEFAULT 0,
                session_count INTEGER NOT NULL DEFAULT 0
            );
        """)
        self._conn.commit()
        # Additive migrations — safe to run every startup (idempotent).
        # ALTER TABLE fails silently if the column already exists.
        for stmt, label in (
            ("ALTER TABLE users ADD COLUMN level INTEGER NOT NULL DEFAULT 1",
             "users.level"),
            ("ALTER TABLE users ADD COLUMN pass_streak INTEGER NOT NULL DEFAULT 0",
             "users.pass_streak"),
            ("ALTER TABLE users ADD COLUMN fail_streak INTEGER NOT NULL DEFAULT 0",
             "users.fail_streak"),
            ("ALTER TABLE daily_plans ADD COLUMN user_id INTEGER REFERENCES users(id)",
             "daily_plans.user_id"),
        ):
            try:
                self._conn.execute(stmt)
                self._conn.commit()
                logger.info("Migration: added '%s' column", label)
            except Exception:
                pass  # Column already exists — nothing to do
        # Legacy DBs had UNIQUE(date) on daily_plans; per-user plans need
        # UNIQUE(date, user_id) instead. Rebuild the table when we detect the
        # old constraint (safe: the copy preserves all existing rows).
        try:
            self._maybe_rebuild_daily_plans()
        except Exception as e:
            logger.warning("daily_plans rebuild skipped: %s", e)
        # Now that user_id exists, create the composite unique index.
        try:
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_daily_plans_date_user ON daily_plans(date, user_id)"
            )
            self._conn.commit()
        except Exception as e:
            logger.warning("daily_plans unique index skipped: %s", e)

    def _maybe_rebuild_daily_plans(self) -> None:
        """Drop legacy UNIQUE(date) constraint by rebuilding the table."""
        assert self._conn is not None
        # Detect the old schema: SQL text still mentions "date TEXT NOT NULL UNIQUE".
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='daily_plans'"
        ).fetchone()
        if not row or not row["sql"]:
            return
        if "date          TEXT NOT NULL UNIQUE" not in row["sql"] \
                and "date TEXT NOT NULL UNIQUE" not in row["sql"]:
            return  # Already on the new schema
        logger.info("Migration: rebuilding daily_plans to drop UNIQUE(date)")
        self._conn.executescript("""
            BEGIN;
            CREATE TABLE daily_plans_new (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                date          TEXT NOT NULL,
                user_id       INTEGER REFERENCES users(id),
                unit_id       INTEGER NOT NULL,
                unit_name     TEXT NOT NULL,
                passed        INTEGER NOT NULL DEFAULT 0,
                session_count INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO daily_plans_new
                (id, date, user_id, unit_id, unit_name, passed, session_count)
            SELECT id, date, user_id, unit_id, unit_name, passed, session_count
              FROM daily_plans;
            DROP TABLE daily_plans;
            ALTER TABLE daily_plans_new RENAME TO daily_plans;
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_daily_plans_date_user ON daily_plans(date, user_id);
            COMMIT;
        """)

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

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Small in-house Levenshtein — avoids adding a dependency for two names."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                curr[j] = min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            prev = curr
        return prev[-1]

    @staticmethod
    def _canonicalize_name(
        name: str,
        known_names: List[str] | None,
        known_aliases: Dict[str, str] | None = None,
    ) -> str:
        """Return the canonical known-name that matches `name`, else `name` unchanged.

        Matching order:
        1. Exact-lowercase hit in ``known_aliases`` (script-independent
           spellings such as "باب" → "Bob").
        2. Any of these against ``known_names`` (case-insensitive):
           - exact equality;
           - one is a prefix of the other and the shorter is ≥ 3 chars;
           - Levenshtein distance ≤ 1 for names of length ≤ 6, ≤ 2 otherwise.

        Only used when a caller passes an explicit ``known_names`` list — never
        applied on a bare ``get_or_create_user(name)`` call, so tests that
        create arbitrary users stay unaffected.
        """
        if not known_names and not known_aliases:
            return name
        raw = name.strip()
        lo = raw.lower()
        if known_aliases and lo in known_aliases:
            return known_aliases[lo]
        if not known_names:
            return raw
        for canonical in known_names:
            can_lo = canonical.strip().lower()
            if lo == can_lo:
                return canonical
            short, long = (lo, can_lo) if len(lo) <= len(can_lo) else (can_lo, lo)
            if len(short) >= 3 and long.startswith(short):
                return canonical
            threshold = 1 if max(len(lo), len(can_lo)) <= 6 else 2
            if SessionDB._levenshtein(lo, can_lo) <= threshold:
                return canonical
        return raw

    def get_or_create_user(
        self,
        name: str,
        known_names: List[str] | None = None,
        known_aliases: Dict[str, str] | None = None,
    ) -> int:
        """Return user id, creating the row if needed.

        Args:
            name:           The name as heard/typed. Whitespace-trimmed on lookup.
            known_names:    Optional list of canonical names (e.g. from
                            ENGLISH_TEACHER_USERS). If provided, the incoming
                            name is snapped to the closest canonical match
                            before insertion, so mis-hearings like "Bobb"
                            collapse into the existing "Bob" row.
            known_aliases:  Optional dict of ``lowercased alias → canonical``
                            entries (e.g. Persian-script spellings such as
                            ``"باب" → "Bob"``).
        """
        assert self._conn is not None
        canonical = self._canonicalize_name(name, known_names, known_aliases)
        name_lower = canonical.strip().lower()
        row = self._conn.execute(
            "SELECT id FROM users WHERE LOWER(name) = ?", (name_lower,)
        ).fetchone()
        if row:
            return int(row["id"])
        cur = self._conn.execute(
            "INSERT INTO users (name) VALUES (?)", (canonical.strip(),)
        )
        self._conn.commit()
        user_id = cur.lastrowid
        assert user_id is not None
        logger.info("Created new user '%s' (id=%d)", canonical, user_id)
        return int(user_id)

    def merge_users(self, source_id: int, target_id: int) -> None:
        """Merge all sessions from `source_id` into `target_id`, then delete source.

        Used by the migration script to collapse duplicate rows created before
        alias normalization was added. Highest stored level wins.
        """
        assert self._conn is not None
        if source_id == target_id:
            return
        src = self._conn.execute(
            "SELECT id, name, level FROM users WHERE id = ?", (source_id,)
        ).fetchone()
        tgt = self._conn.execute(
            "SELECT id, name, level FROM users WHERE id = ?", (target_id,)
        ).fetchone()
        if not src or not tgt:
            raise ValueError(f"merge_users: source={source_id} target={target_id} — one is missing")
        # Re-point every session owned by source to target
        self._conn.execute(
            "UPDATE sessions SET user_id = ? WHERE user_id = ?", (target_id, source_id)
        )
        # Keep the higher level between the two
        merged_level = max(int(src["level"] or 1), int(tgt["level"] or 1))
        self._conn.execute(
            "UPDATE users SET level = ? WHERE id = ?", (merged_level, target_id)
        )
        # Delete the source row
        self._conn.execute("DELETE FROM users WHERE id = ?", (source_id,))
        self._conn.commit()
        logger.info(
            "Merged user '%s' (id=%d) into '%s' (id=%d), level=%d",
            src["name"], source_id, tgt["name"], target_id, merged_level,
        )

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

    # Number of consecutive passes at the current level required to level up,
    # and consecutive fails to drop a level. Kept low so real-world use
    # progresses at a felt pace.
    PASS_TO_LEVEL_UP = 3
    FAIL_TO_LEVEL_DOWN = 2

    def record_session_result(
        self, user_id: int, passed: bool
    ) -> tuple[int, bool]:
        """Update the user's pass/fail streaks and auto-adjust level.

        Returns ``(new_level, level_changed)``. Called by the summarizer at
        session end. Users without enough evidence stay at their current level.
        """
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT level, pass_streak, fail_streak FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return (1, False)

        level = max(1, min(3, int(row["level"] or 1)))
        pass_streak = int(row["pass_streak"] or 0)
        fail_streak = int(row["fail_streak"] or 0)
        original_level = level
        if passed:
            pass_streak += 1
            fail_streak = 0
            if pass_streak >= self.PASS_TO_LEVEL_UP and level < 3:
                level += 1
                pass_streak = 0
                logger.info("User id=%d levelled UP to %d", user_id, level)
        else:
            fail_streak += 1
            pass_streak = 0
            if fail_streak >= self.FAIL_TO_LEVEL_DOWN and level > 1:
                level -= 1
                fail_streak = 0
                logger.info("User id=%d levelled DOWN to %d", user_id, level)

        self._conn.execute(
            "UPDATE users SET level = ?, pass_streak = ?, fail_streak = ? WHERE id = ?",
            (level, pass_streak, fail_streak, user_id),
        )
        self._conn.commit()
        return (level, level != original_level)
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

        When ``user_id`` is None (speaker not yet identified) this returns None
        instead of a globally-mixed recap — otherwise the most-recent user's
        summaries would leak into the prompt and bias identification.
        """
        assert self._conn is not None
        if user_id is None:
            return None
        _STUB_SUMMARIES = frozenset([
            "Profile switched",
            "جلسه بسیار کوتاه بود — موضوع خاصی تمرین نشد.",
        ])
        conditions = [
            "summary IS NOT NULL",
            "length(trim(summary)) > 20",
            "user_id = ?",
        ]
        params: list = [user_id]

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

    @staticmethod
    def _user_id_sql(user_id: Optional[int]) -> tuple[str, tuple]:
        """SQLite treats NULL as distinct in WHERE `=` — return a matcher pair."""
        if user_id is None:
            return ("user_id IS NULL", ())
        return ("user_id = ?", (user_id,))

    def get_or_create_daily_plan(
        self,
        date: str,
        user_level: int = 1,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return today's unit plan for ``user_id``, creating it if missing.

        Plans are scoped by ``(date, user_id)`` so different students can be
        working through different units on the same day. ``user_id=None``
        preserves the legacy shared "anonymous" plan for pre-identification
        sessions.

        The unit is chosen within the user's level band (LEVEL_UNIT_RANGES):
        - If the user's last previous day was passed → advance to the next
          unit within the same band, wrapping when the band's last unit was
          already passed.
        - If not passed (or no history for this user) → repeat the previous
          unit, or snap into the current level's band.
        - Very first day ever for this user → start at the band's first unit.
        """
        assert self._conn is not None

        u_where, u_params = self._user_id_sql(user_id)
        row = self._conn.execute(
            f"SELECT * FROM daily_plans WHERE date = ? AND {u_where}",
            (date, *u_params),
        ).fetchone()
        if row:
            return dict(row)

        lo, hi = LEVEL_UNIT_RANGES.get(int(user_level), LEVEL_UNIT_RANGES[1])

        # Determine unit for today based on this user's last completed day
        unit_id = lo
        last = self._conn.execute(
            f"SELECT unit_id, passed FROM daily_plans "
            f"WHERE date < ? AND {u_where} ORDER BY date DESC LIMIT 1",
            (date, *u_params),
        ).fetchone()
        if last:
            last_uid = int(last["unit_id"])
            if last["passed"]:
                # Advance one unit; stay inside the band, wrap at the top
                if lo <= last_uid < hi:
                    unit_id = last_uid + 1
                elif last_uid == hi:
                    unit_id = lo
                else:
                    # Previous plan was outside this user's band → start fresh at band low
                    unit_id = lo
            else:
                # Repeat, but only if the previous unit is inside this band
                unit_id = last_uid if lo <= last_uid <= hi else lo

        unit_name = next(
            (u["unit_name"] for u in CURRICULUM if u["unit_id"] == unit_id), "احوالپرسی پایه"
        )
        self._conn.execute(
            "INSERT INTO daily_plans (date, user_id, unit_id, unit_name) VALUES (?, ?, ?, ?)",
            (date, user_id, unit_id, unit_name),
        )
        self._conn.commit()
        logger.info(
            "Daily plan created for %s (user_id=%s) → unit %d (%s)",
            date, user_id, unit_id, unit_name,
        )
        return dict(self._conn.execute(
            f"SELECT * FROM daily_plans WHERE date = ? AND {u_where}",
            (date, *u_params),
        ).fetchone())

    def increment_daily_session_count(
        self, date: str, user_id: Optional[int] = None
    ) -> None:
        """Increment the session count for ``(date, user_id)``."""
        assert self._conn is not None
        u_where, u_params = self._user_id_sql(user_id)
        self._conn.execute(
            f"UPDATE daily_plans SET session_count = session_count + 1 "
            f"WHERE date = ? AND {u_where}",
            (date, *u_params),
        )
        self._conn.commit()

    def mark_daily_plan_result(
        self, date: str, passed: bool, user_id: Optional[int] = None
    ) -> None:
        """Record whether ``user_id`` passed their unit on ``date``.

        A day is only marked 'passed' if it hasn't been passed before,
        so multiple sessions on the same day cannot downgrade a pass.
        """
        assert self._conn is not None
        u_where, u_params = self._user_id_sql(user_id)
        if passed:
            self._conn.execute(
                f"UPDATE daily_plans SET passed = 1 WHERE date = ? AND {u_where}",
                (date, *u_params),
            )
        else:
            # Only update to 0 if it hasn't already been passed today
            self._conn.execute(
                f"UPDATE daily_plans SET passed = 0 "
                f"WHERE date = ? AND {u_where} AND passed = 0",
                (date, *u_params),
            )
        self._conn.commit()
        logger.info(
            "Daily plan %s (user_id=%s) → %s",
            date, user_id, "PASSED ✓" if passed else "not passed yet",
        )

    def get_daily_plan_for_prompt(
        self,
        date: str,
        user_level: int | None = None,
        user_id: Optional[int] = None,
    ) -> str:
        """Return a formatted string for injection into the system prompt.

        When ``user_level`` is given, the stored unit is clamped into that
        level's band so a session belonging to a higher-level user is not
        forced to work through a beginner unit that was set earlier that day.
        ``user_id`` selects that user's plan; ``None`` matches the legacy
        anonymous row for pre-identification sessions.
        """
        assert self._conn is not None
        u_where, u_params = self._user_id_sql(user_id)
        row = self._conn.execute(
            f"SELECT unit_id, unit_name, session_count, passed FROM daily_plans "
            f"WHERE date = ? AND {u_where}",
            (date, *u_params),
        ).fetchone()
        if not row:
            return ""

        unit_id   = int(row["unit_id"])
        unit_name = row["unit_name"]
        session_count = row["session_count"]

        if user_level is not None:
            lo, hi = LEVEL_UNIT_RANGES.get(int(user_level), LEVEL_UNIT_RANGES[1])
            if not (lo <= unit_id <= hi):
                unit_id = lo
                unit_name = next(
                    (u["unit_name"] for u in CURRICULUM if u["unit_id"] == unit_id),
                    unit_name,
                )

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

        n_phrases = len(phrases)
        return (
            f"## برنامه درس امروز\n"
            f"**واحد {unit_id} — {unit_name}**\n"
            f"عبارات هدف ({n_phrases} عبارت): {phrases_str}\n\n"
            f"🔒 **قفل روی این واحد.** کل جلسه امروز فقط با این {n_phrases} عبارت "
            f"کار می‌کنی. اجازه نداری موضوعات دیگر (آب‌وهوا، غذا، خرید، سن، شهر، …) "
            f"را وارد جلسه کنی مگر اینکه در همان لیست بالا باشند.\n"
            f"🔁 **چرخش سناریو، نه چرخش موضوع.** همین عبارات را در سناریوهای مختلف "
            f"(صبح در داروخانه، برخورد با همسایه، ملاقات دوست قدیمی، تلفنی، …) "
            f"تمرین کن — هر عبارت باید در حداقل ۳ سناریوی متفاوت استفاده شود قبل "
            f"از اینکه فرض کنی کاربر آن را یاد گرفته.\n"
            f"⛔ اگر کاربر خودش موضوع دیگری را مطرح کرد، کوتاه به فارسی جواب بده و "
            f"با یک سناریوی جدید از **همین {n_phrases} عبارت** برگرد — هرگز "
            f"«می‌خوای درباره X صحبت کنیم؟» با موضوع خارج از این لیست نپرس.\n"
            f"🚪 جلسه فقط با خداحافظی صریح کاربر تمام می‌شود، نه با تمام شدن "
            f"عبارات — این عبارات تمام‌نشدنی‌اند، فقط سناریو عوض می‌کنی.{repeat_note}"
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

