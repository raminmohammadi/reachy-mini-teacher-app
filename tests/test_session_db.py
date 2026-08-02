"""Comprehensive tests for SessionDB – the SQLite persistence layer."""
from __future__ import annotations
import datetime
from pathlib import Path
import pytest
from reachy_mini_teacher_app.session_db import SessionDB, CURRICULUM


@pytest.fixture()
def db(tmp_path: Path) -> SessionDB:
    _db = SessionDB(db_path=tmp_path / "test.db")
    yield _db
    _db.close()


class TestUserManagement:
    def test_create_user(self, db):
        uid = db.get_or_create_user("Alice")
        assert isinstance(uid, int) and uid > 0

    def test_get_existing_user(self, db):
        assert db.get_or_create_user("Bob") == db.get_or_create_user("Bob")

    def test_case_insensitive_lookup(self, db):
        uid = db.get_or_create_user("Charlie")
        assert uid == db.get_or_create_user("charlie") == db.get_or_create_user("CHARLIE")

    def test_whitespace_stripped(self, db):
        assert db.get_or_create_user("  Dana  ") == db.get_or_create_user("Dana")

    def test_list_users_empty(self, db):
        assert db.list_users() == []

    def test_list_users(self, db):
        db.get_or_create_user("Alice")
        db.get_or_create_user("Bob")
        assert len(db.list_users()) == 2
        assert {u["name"] for u in db.list_users()} == {"Alice", "Bob"}

    def test_farsi_name(self, db):
        uid = db.get_or_create_user("مریم")
        assert uid > 0
        assert uid == db.get_or_create_user("مریم")


class TestSessionManagement:
    def test_start_session_no_user(self, db):
        assert db.start_session() > 0

    def test_start_session_with_user(self, db):
        uid = db.get_or_create_user("Eve")
        assert db.start_session(user_id=uid) > 0

    def test_end_session_with_summary(self, db):
        uid = db.get_or_create_user("Tester")
        sid = db.start_session(user_id=uid)
        db.end_session(sid, summary="Test summary")
        assert db.get_latest_summary(uid) == "Test summary"

    def test_assign_session_user(self, db):
        sid = db.start_session()
        uid = db.get_or_create_user("Frank")
        db.assign_session_user(sid, uid)
        assert db.get_user_session_count(uid) == 1

    def test_multiple_sessions_per_user(self, db):
        uid = db.get_or_create_user("Grace")
        for _ in range(3):
            db.start_session(user_id=uid)
        assert db.get_user_session_count(uid) == 3


class TestMessages:
    def test_add_and_get(self, db):
        sid = db.start_session()
        db.add_message(sid, "user", "Hello")
        db.add_message(sid, "assistant", "Hi!")
        msgs = db.get_session_messages(sid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user" and msgs[0]["content"] == "Hello"

    def test_empty_message_ignored(self, db):
        sid = db.start_session()
        db.add_message(sid, "user", "")
        db.add_message(sid, "user", "   ")
        assert db.get_session_messages(sid) == []

    def test_whitespace_stripped(self, db):
        sid = db.start_session()
        db.add_message(sid, "user", "  hello  ")
        assert db.get_session_messages(sid)[0]["content"] == "hello"

    def test_farsi_content(self, db):
        sid = db.start_session()
        db.add_message(sid, "assistant", "سلام! حالتون چطوره؟")
        assert db.get_session_messages(sid)[0]["content"] == "سلام! حالتون چطوره؟"

    def test_ordering(self, db):
        sid = db.start_session()
        for i in range(5):
            db.add_message(sid, "user", f"msg-{i}")
        assert [m["content"] for m in db.get_session_messages(sid)] == [f"msg-{i}" for i in range(5)]


class TestSummariesAndRecap:
    def test_no_summary(self, db):
        uid = db.get_or_create_user("Heidi")
        assert db.get_latest_summary(uid) is None

    def test_latest_summary(self, db):
        uid = db.get_or_create_user("Ivan")
        s1 = db.start_session(user_id=uid)
        db.end_session(s1, summary="S1")
        s2 = db.start_session(user_id=uid)
        db.end_session(s2, summary="S2")
        assert db.get_latest_summary(uid) == "S2"

    def test_latest_summary_skips_null(self, db):
        uid = db.get_or_create_user("Judy")
        s1 = db.start_session(user_id=uid)
        db.end_session(s1, summary="First")
        s2 = db.start_session(user_id=uid)
        db.end_session(s2)
        assert db.get_latest_summary(uid) == "First"

    def test_recap_first_session_returns_none(self, db):
        uid = db.get_or_create_user("Karl")
        db.start_session(user_id=uid)
        assert db.build_recap_for_user(uid) is None

    def test_recap_with_history(self, db):
        uid = db.get_or_create_user("مریم")
        s1 = db.start_session(user_id=uid)
        db.end_session(s1, summary="یادگرفت: Hello")
        db.start_session(user_id=uid)
        recap = db.build_recap_for_user(uid)
        assert recap and "جلسه شماره 2" in recap and "Hello" in recap

    def test_recap_multiple_sessions_no_summary(self, db):
        uid = db.get_or_create_user("Leo")
        db.end_session(db.start_session(user_id=uid))
        db.start_session(user_id=uid)
        recap = db.build_recap_for_user(uid)
        assert recap and "جلسه شماره 2" in recap

    def test_session_count(self, db):
        uid = db.get_or_create_user("Mallory")
        assert db.get_user_session_count(uid) == 0
        db.start_session(user_id=uid)
        assert db.get_user_session_count(uid) == 1


class TestConnectionLifecycle:
    def test_persistence(self, tmp_path):
        db1 = SessionDB(db_path=tmp_path / "p.db")
        uid = db1.get_or_create_user("Persist")
        db1.close()
        db2 = SessionDB(db_path=tmp_path / "p.db")
        assert db2.get_or_create_user("Persist") == uid
        db2.close()

    def test_double_close(self, db):
        db.close()
        db.close()

    def test_default_path_creates_directory(self, tmp_path, monkeypatch):
        fake = tmp_path / "fakehome" / ".reachy_mini" / "sessions.db"
        monkeypatch.setattr("reachy_mini_teacher_app.session_db._DEFAULT_DB_PATH", fake)
        _db = SessionDB()
        assert fake.parent.exists()
        _db.close()


class TestDailyPlan:
    """Tests for curriculum scheduling: get_or_create_daily_plan, mark/increment, prompt text."""

    def _yesterday(self):
        return (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    def _today(self):
        return datetime.date.today().isoformat()

    def _tomorrow(self):
        return (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    def test_first_ever_session_starts_at_unit_1(self, db):
        plan = db.get_or_create_daily_plan(self._today())
        assert plan["unit_id"] == 1

    def test_idempotent_same_day(self, db):
        today = self._today()
        p1 = db.get_or_create_daily_plan(today)
        p2 = db.get_or_create_daily_plan(today)
        assert p1["unit_id"] == p2["unit_id"]

    def test_advance_to_next_unit_after_pass(self, db):
        db.get_or_create_daily_plan(self._yesterday())
        db.mark_daily_plan_result(self._yesterday(), passed=True)
        plan = db.get_or_create_daily_plan(self._today())
        assert plan["unit_id"] == 2

    def test_repeat_same_unit_after_fail(self, db):
        db.get_or_create_daily_plan(self._yesterday())
        db.mark_daily_plan_result(self._yesterday(), passed=False)
        plan = db.get_or_create_daily_plan(self._today())
        assert plan["unit_id"] == 1

    def test_wrap_around_after_unit_7(self, db):
        # Simulate passing unit 7
        db.get_or_create_daily_plan(self._yesterday())
        # Force unit 7 directly via internal update
        db._conn.execute(
            "UPDATE daily_plans SET unit_id=7, unit_name='آب‌وهوا', passed=1 WHERE date=?",
            (self._yesterday(),)
        )
        db._conn.commit()
        plan = db.get_or_create_daily_plan(self._today())
        assert plan["unit_id"] == 1  # wraps back to 1

    def test_pass_cannot_be_downgraded_to_fail(self, db):
        today = self._today()
        db.get_or_create_daily_plan(today)
        db.mark_daily_plan_result(today, passed=True)
        db.mark_daily_plan_result(today, passed=False)  # should be ignored
        plan = db.get_or_create_daily_plan(today)
        assert plan["passed"] == 1

    def test_increment_session_count(self, db):
        today = self._today()
        db.get_or_create_daily_plan(today)
        db.increment_daily_session_count(today)
        db.increment_daily_session_count(today)
        plan = db.get_or_create_daily_plan(today)
        assert plan["session_count"] == 2

    def test_prompt_text_contains_unit_name_and_phrases(self, db):
        today = self._today()
        db.get_or_create_daily_plan(today)
        text = db.get_daily_plan_for_prompt(today)
        assert "واحد 1" in text
        assert "احوالپرسی" in text
        assert "Hello" in text

    def test_prompt_text_contains_unit_lock_language(self, db):
        today = self._today()
        db.get_or_create_daily_plan(today)
        text = db.get_daily_plan_for_prompt(today)
        # Explicit lock wording preventing topic drift
        assert "قفل" in text
        assert "چرخش سناریو" in text
        # Forbidden topics must be named so Zizi cannot rationalize drifting
        assert "آب‌وهوا" in text
        assert "غذا" in text
        assert "خرید" in text
        # Session must only end on user's explicit goodbye
        assert "خداحافظی" in text

    def test_prompt_text_contains_return_note_after_increment(self, db):
        today = self._today()
        db.get_or_create_daily_plan(today)
        db.increment_daily_session_count(today)
        text = db.get_daily_plan_for_prompt(today)
        assert "جلسه" in text  # mentions previous session count

    def test_prompt_text_empty_for_missing_date(self, db):
        assert db.get_daily_plan_for_prompt("1900-01-01") == ""

    def test_curriculum_has_twenty_one_units_across_three_levels(self):
        # 7 beginner + 7 intermediate + 7 advanced = 21 units, ids 1..21
        assert len(CURRICULUM) == 21
        ids = [u["unit_id"] for u in CURRICULUM]
        assert ids == list(range(1, 22))

    def test_all_units_have_required_keys(self):
        for unit in CURRICULUM:
            assert "unit_id" in unit
            assert "unit_name" in unit
            assert isinstance(unit["phrases"], list) and len(unit["phrases"]) > 0


class TestNameMemory:
    """Tests for get_most_recent_user_name and unlink_session_user."""

    def test_returns_none_with_no_sessions(self, db):
        assert db.get_most_recent_user_name() is None

    def test_returns_none_for_anonymous_sessions(self, db):
        db.start_session()  # no user linked
        assert db.get_most_recent_user_name() is None

    def test_returns_name_after_linking(self, db):
        sid = db.start_session()
        uid = db.get_or_create_user("Ali")
        db.assign_session_user(sid, uid)
        assert db.get_most_recent_user_name() == "Ali"

    def test_returns_most_recent_linked_name(self, db):
        s1 = db.start_session()
        db.assign_session_user(s1, db.get_or_create_user("Alice"))
        s2 = db.start_session()
        db.assign_session_user(s2, db.get_or_create_user("Bob"))
        assert db.get_most_recent_user_name() == "Bob"

    def test_unlink_removes_association(self, db):
        sid = db.start_session()
        uid = db.get_or_create_user("مریم")
        db.assign_session_user(sid, uid)
        assert db.get_most_recent_user_name() == "مریم"
        db.unlink_session_user(sid)
        # No other linked session → should return None
        assert db.get_most_recent_user_name() is None

    def test_unlink_falls_back_to_previous_user(self, db):
        s1 = db.start_session()
        db.assign_session_user(s1, db.get_or_create_user("Alice"))
        s2 = db.start_session()
        db.assign_session_user(s2, db.get_or_create_user("Bob"))
        db.unlink_session_user(s2)  # Bob leaves mid-session
        # Should fall back to Alice (the previous linked session)
        assert db.get_most_recent_user_name() == "Alice"

    def test_farsi_name_stored_and_retrieved(self, db):
        sid = db.start_session()
        db.assign_session_user(sid, db.get_or_create_user("علی"))
        assert db.get_most_recent_user_name() == "علی"



class TestCanonicalizeAndAliases:
    """Alias/known_names resolution used to collapse duplicate rows."""

    def test_known_names_collapses_misspelling(self, db):
        uid = db.get_or_create_user("Bob", known_names=["Bob", "Alice"])
        # "Bobb" is a common mishearing; fuzzy match should snap to Bob.
        assert db.get_or_create_user("Bobb", known_names=["Bob", "Alice"]) == uid

    def test_alias_map_snaps_script_variant(self, db):
        aliases = {"باب": "Bob", "bab": "Bob"}
        uid = db.get_or_create_user(
            "Bob", known_names=["Bob"], known_aliases=aliases,
        )
        # Persian-script alias must resolve to the same canonical row.
        assert db.get_or_create_user(
            "باب", known_names=["Bob"], known_aliases=aliases,
        ) == uid
        # Alias lookup is case-insensitive.
        assert db.get_or_create_user(
            "BAB", known_names=["Bob"], known_aliases=aliases,
        ) == uid

    def test_no_known_names_preserves_original(self, db):
        # Without known_names or aliases, near-matches stay separate.
        a = db.get_or_create_user("Carol")
        b = db.get_or_create_user("Carola")
        assert a != b


class TestMergeUsers:
    def test_merge_moves_sessions(self, db):
        src = db.get_or_create_user("Bobb")
        tgt = db.get_or_create_user("Bob")
        s = db.start_session(user_id=src)
        db.merge_users(src, tgt)
        # Source row is gone; session now belongs to target.
        assert all(u["id"] != src for u in db.list_users())
        assert db.get_user_name_for_session(s) == "Bob"


class TestAutoLeveling:
    def test_level_up_after_consecutive_passes(self, db):
        uid = db.get_or_create_user("Learner")
        for _ in range(db.PASS_TO_LEVEL_UP):
            level, changed = db.record_session_result(uid, passed=True)
        assert changed is True
        assert level == 2

    def test_level_down_after_consecutive_fails(self, db):
        uid = db.get_or_create_user("Struggler")
        db.set_user_level("Struggler", 2)
        for _ in range(db.FAIL_TO_LEVEL_DOWN):
            level, changed = db.record_session_result(uid, passed=False)
        assert changed is True
        assert level == 1

    def test_pass_resets_fail_streak(self, db):
        uid = db.get_or_create_user("Mixed")
        db.set_user_level("Mixed", 2)
        db.record_session_result(uid, passed=False)  # fail_streak=1
        db.record_session_result(uid, passed=True)   # resets fail_streak
        # One more fail should NOT drop the level (streak was reset).
        _, changed = db.record_session_result(uid, passed=False)
        assert changed is False
        assert db.get_user_level("Mixed") == 2

    def test_cannot_exceed_level_3(self, db):
        uid = db.get_or_create_user("Top")
        db.set_user_level("Top", 3)
        for _ in range(db.PASS_TO_LEVEL_UP * 2):
            level, _ = db.record_session_result(uid, passed=True)
        assert level == 3

    def test_cannot_go_below_level_1(self, db):
        uid = db.get_or_create_user("Bottom")
        for _ in range(db.FAIL_TO_LEVEL_DOWN * 2):
            level, _ = db.record_session_result(uid, passed=False)
        assert level == 1



class TestPerUserDailyPlans:
    """Two users must have independent daily plans on the same date."""

    def _today(self):
        return datetime.date.today().isoformat()

    def _yesterday(self):
        return (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    def test_two_users_get_independent_plans(self, db):
        today = self._today()
        alice = db.get_or_create_user("Alice")
        bob = db.get_or_create_user("Bob")
        # Alice passes yesterday → advances to unit 2 today.
        db.get_or_create_daily_plan(self._yesterday(), user_id=alice)
        db.mark_daily_plan_result(self._yesterday(), passed=True, user_id=alice)
        alice_plan = db.get_or_create_daily_plan(today, user_id=alice)
        # Bob has no history → starts at unit 1 (band low).
        bob_plan = db.get_or_create_daily_plan(today, user_id=bob)
        assert alice_plan["unit_id"] == 2
        assert bob_plan["unit_id"] == 1

    def test_mark_result_scoped_to_user(self, db):
        today = self._today()
        alice = db.get_or_create_user("Alice")
        bob = db.get_or_create_user("Bob")
        db.get_or_create_daily_plan(today, user_id=alice)
        db.get_or_create_daily_plan(today, user_id=bob)
        db.mark_daily_plan_result(today, passed=True, user_id=alice)
        # Bob's row must NOT be marked passed as a side effect.
        assert db.get_or_create_daily_plan(today, user_id=alice)["passed"] == 1
        assert db.get_or_create_daily_plan(today, user_id=bob)["passed"] == 0

    def test_session_count_scoped_to_user(self, db):
        today = self._today()
        alice = db.get_or_create_user("Alice")
        bob = db.get_or_create_user("Bob")
        db.get_or_create_daily_plan(today, user_id=alice)
        db.get_or_create_daily_plan(today, user_id=bob)
        db.increment_daily_session_count(today, user_id=alice)
        db.increment_daily_session_count(today, user_id=alice)
        db.increment_daily_session_count(today, user_id=bob)
        assert db.get_or_create_daily_plan(today, user_id=alice)["session_count"] == 2
        assert db.get_or_create_daily_plan(today, user_id=bob)["session_count"] == 1

    def test_anonymous_plan_isolated_from_user_plans(self, db):
        # Legacy pre-identification plans (user_id=None) must not leak into
        # a real user's plan on the same date.
        today = self._today()
        db.get_or_create_daily_plan(today)  # anonymous
        db.mark_daily_plan_result(today, passed=True)  # anonymous passes
        alice = db.get_or_create_user("Alice")
        alice_plan = db.get_or_create_daily_plan(today, user_id=alice)
        assert alice_plan["passed"] == 0
        # Anonymous plan is unaffected by user activity.
        db.mark_daily_plan_result(today, passed=False, user_id=alice)
        assert db.get_or_create_daily_plan(today)["passed"] == 1

    def test_prompt_text_scoped_to_user(self, db):
        today = self._today()
        alice = db.get_or_create_user("Alice")
        db.get_or_create_daily_plan(today, user_id=alice)
        # Another user's plan on the same day shouldn't appear in Alice's prompt.
        bob = db.get_or_create_user("Bob")
        db.get_or_create_daily_plan(self._yesterday(), user_id=bob)
        db.mark_daily_plan_result(self._yesterday(), passed=True, user_id=bob)
        db.get_or_create_daily_plan(today, user_id=bob)  # bob on unit 2
        text = db.get_daily_plan_for_prompt(today, user_id=alice)
        assert "واحد 1" in text  # alice's unit, not bob's unit 2
