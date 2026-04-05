"""Tests for user-identity tools: remember_user_name and switch_user."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_teacher_app.session_db import SessionDB
from reachy_mini_teacher_app.tools.remember_user_name import RememberUserName
from reachy_mini_teacher_app.tools.switch_user import SwitchUser


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path: Path) -> SessionDB:
    _db = SessionDB(db_path=tmp_path / "tools_test.db")
    yield _db
    _db.close()


@pytest.fixture()
def deps(db):
    """Minimal ToolDependencies mock wired to the real test DB."""
    sid = db.start_session()
    mock = MagicMock()
    mock.session_db = db
    mock.session_state = {"session_id": sid}
    mock._session_id = sid
    return mock


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── RememberUserName ─────────────────────────────────────────────────────────

class TestRememberUserName:
    def test_stores_name_in_db(self, db, deps):
        result = run(RememberUserName()(deps, name="Sara"))
        assert result["status"] == "ok"
        assert db.get_most_recent_user_name() == "Sara"

    def test_farsi_name(self, db, deps):
        result = run(RememberUserName()(deps, name="مریم"))
        assert result["status"] == "ok"
        assert db.get_most_recent_user_name() == "مریم"

    def test_empty_name_ignored(self, db, deps):
        result = run(RememberUserName()(deps, name=""))
        assert result["status"] == "ignored"
        assert db.get_most_recent_user_name() is None

    def test_whitespace_only_ignored(self, db, deps):
        result = run(RememberUserName()(deps, name="   "))
        assert result["status"] == "ignored"

    def test_no_db_graceful(self):
        mock = MagicMock()
        mock.session_db = None
        mock.session_state = {"session_id": 1}
        result = run(RememberUserName()(mock, name="Ali"))
        assert result["status"] == "ignored"

    def test_no_session_id_still_creates_user(self, db):
        mock = MagicMock()
        mock.session_db = db
        mock.session_state = {"session_id": None}
        result = run(RememberUserName()(mock, name="NoSession"))
        # User created in DB but not linked to a session
        assert result["status"] == "ok"
        uid = db.get_or_create_user("NoSession")
        assert uid > 0

    def test_calling_twice_keeps_latest_name(self, db, deps):
        run(RememberUserName()(deps, name="Alice"))
        # Start a new session and link a different name
        sid2 = db.start_session()
        deps.session_state["session_id"] = sid2
        run(RememberUserName()(deps, name="Bob"))
        assert db.get_most_recent_user_name() == "Bob"

    def test_same_name_twice_idempotent(self, db, deps):
        run(RememberUserName()(deps, name="Ali"))
        run(RememberUserName()(deps, name="Ali"))
        assert db.get_most_recent_user_name() == "Ali"


# ── SwitchUser ───────────────────────────────────────────────────────────────

class TestSwitchUser:
    def test_unlinks_current_session(self, db, deps):
        # First link a user
        run(RememberUserName()(deps, name="Alice"))
        assert db.get_most_recent_user_name() == "Alice"

        # Switch user — should unlink Alice from this session
        result = run(SwitchUser()(deps))
        assert result["status"] == "ok"
        assert db.get_most_recent_user_name() is None

    def test_after_switch_new_name_remembered(self, db, deps):
        run(RememberUserName()(deps, name="Alice"))
        run(SwitchUser()(deps))
        run(RememberUserName()(deps, name="Bob"))
        assert db.get_most_recent_user_name() == "Bob"

    def test_falls_back_to_previous_linked_session(self, db):
        # Session 1: Alice
        s1 = db.start_session()
        db.assign_session_user(s1, db.get_or_create_user("Alice"))

        # Session 2: Bob logs in then switches
        s2 = db.start_session()
        deps2 = MagicMock()
        deps2.session_db = db
        deps2.session_state = {"session_id": s2}
        run(RememberUserName()(deps2, name="Bob"))
        run(SwitchUser()(deps2))

        # Most recent linked session is now s1 (Alice)
        assert db.get_most_recent_user_name() == "Alice"

    def test_no_db_graceful(self):
        mock = MagicMock()
        mock.session_db = None
        mock.session_state = {"session_id": 1}
        result = run(SwitchUser()(mock))
        assert result["status"] == "ignored"

    def test_no_session_id_still_ok(self, db):
        mock = MagicMock()
        mock.session_db = db
        mock.session_state = {"session_id": None}
        result = run(SwitchUser()(mock))
        assert result["status"] == "ok"
