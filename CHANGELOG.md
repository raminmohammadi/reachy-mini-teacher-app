# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Per-user daily lesson plans.** `daily_plans` now has a `user_id`
  column with a unique `(date, user_id)` index, so two users on the same
  day get independent curricula and pass/fail streaks. Legacy rows are
  preserved with `user_id = NULL`.
- **Alias-aware user identification.** `ENGLISH_TEACHER_USERS` accepts a
  `Name/alias1/alias2:gender` syntax; `SessionDB._canonicalize_name`
  collapses mis-hearings (Levenshtein ≤ 1–2) and Persian-script variants
  onto the canonical row via a lowercased alias map.
- **`SessionDB.merge_users()`** and the `scripts/migrate_db.py` utility
  for merging duplicate user rows and backfilling `sessions.user_id`.
- **Unit-locked prompt block.** `get_daily_plan_for_prompt` now injects
  explicit "lock" language that forbids the model from introducing
  vocabulary outside the current unit's target list, and requires
  scenario rotation instead of topic hopping.
- **Unit-aware session summariser.** `session_summarizer` receives the
  current unit's target phrases and only grants the *«قبول»* verdict
  when the learner uses ≥ 3 of them across ≥ 2 distinct scenarios; a
  new "outside-unit drift" field surfaces teacher-driven derailment.
- **YOLO-based head tracking** for the face-follow behaviour
  (already on the `new_head_tracking` branch).
- **`CHANGELOG.md`** (this file).

### Changed
- **English-teacher instructions** rewritten to enforce scenario
  rotation, ban Zizi from ending sessions, and gate the `check_weather`
  tool to the weather unit only.
- **Genericised examples** across docstrings, comments, and test
  fixtures — real user names (`Khosro`, `Shadi`, `Khosrow`, `خسرو`,
  `شادی`, `Hosro`) replaced with `Alice`/`Bob`/`باب`/`آلیس` so the repo
  can be published without exposing personal identifiers.
- **Portable launcher scripts.** `scripts/launch_teacher_app.sh` and
  `scripts/stop_teacher_app.sh` derive `REPO_DIR` from
  `$BASH_SOURCE` instead of a hard-coded `/home/<username>/…` path.
- **`.desktop` file templates** now carry an `INSTALL_DIR/...`
  placeholder `Exec=` line (rewritten at install time by
  `scripts/install_launcher.sh`).
- **Prompt-loading pipeline** (`prompts.py`) now injects
  `{{user_name}}` and the daily-plan block alongside the existing
  `{{previous_recap}}` placeholder.
- **Gemini and OpenAI handlers** buffer streamed transcript fragments
  and persist one merged DB message per turn, so the dashboard no
  longer floods with dozens of one-word bubbles.

### Fixed
- Session-continuation and identification bias that mis-attributed one
  user's transcript to another (root cause: no `user_id` on daily plans
  and no alias/canonicalisation on user lookup).
- Premature session endings by Zizi in response to garbled or non-Farsi
  ASR output (instructions now require asking for clarification).

### Security
- Verified tracked files contain no API keys, tokens, or hard-coded
  credentials. `.env` and `sessions.db` remain gitignored.

## [1.0.0] - 2026-04-04

Initial tagged release: Gemini Live voice teacher on Reachy Mini with a
local fallback pipeline (Faster-Whisper STT + Ollama LLM + Kokoro TTS),
FastAPI dashboard, session database, and a first cut of the English
teaching curriculum.
