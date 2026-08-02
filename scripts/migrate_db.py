#!/usr/bin/env python3
"""One-off data migration for the Reachy Mini Teacher App session DB.

What it does (in this order):
  1. Detect duplicate user rows using the same fuzzy rules as
     ``SessionDB._canonicalize_name`` and merge them into the canonical
     name from ``ENGLISH_TEACHER_USERS``. Higher user id sources are
     merged into the lower canonical target.
  2. Optionally back-fill ``sessions.user_id`` for anonymous rows whose
     stored summary starts with ``کاربر: <name>``.

By default this runs in dry-run mode and only prints what it would do.
Pass ``--apply`` to actually write changes. It refuses to run while the
DB is locked by the live app (WAL will still make it *readable*, but
writes race the live process).

Usage:
    python3 scripts/migrate_db.py               # dry run
    python3 scripts/migrate_db.py --apply       # perform the merge
    python3 scripts/migrate_db.py --apply --backfill-anonymous
"""
from __future__ import annotations

import argparse
import shutil
import sys
import re
from datetime import datetime
from pathlib import Path

# Make ``reachy_mini_teacher_app`` importable when invoked from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reachy_mini_teacher_app.config import config  # noqa: E402
from reachy_mini_teacher_app.session_db import SessionDB  # noqa: E402


def _dupes_to_merge(
    db: SessionDB,
    canonical_names: list[str],
    aliases: dict[str, str],
) -> list[tuple[int, int]]:
    """Return [(source_id, target_id), …] pairs to merge."""
    users = db.list_users()
    # Group user ids by canonical name (case-insensitive fuzzy match)
    by_canonical: dict[str, list[dict]] = {}
    for u in users:
        canon = SessionDB._canonicalize_name(u["name"], canonical_names, aliases)
        by_canonical.setdefault(canon, []).append(u)

    pairs: list[tuple[int, int]] = []
    for canon, members in by_canonical.items():
        if len(members) < 2:
            continue
        # Target = the row whose name matches canon exactly (case-insensitive),
        # else the oldest (lowest id)
        exact = [m for m in members if m["name"].strip().lower() == canon.strip().lower()]
        target = exact[0] if exact else min(members, key=lambda m: m["id"])
        for m in members:
            if m["id"] != target["id"]:
                pairs.append((m["id"], target["id"]))
    return pairs


_KARBAR_LINE = re.compile(r"^\s*کاربر\s*[:：]\s*(.+?)\s*$")


def _guess_user_from_summary(summary: str) -> str | None:
    for line in summary.splitlines():
        m = _KARBAR_LINE.match(line)
        if m:
            name = m.group(1).strip()
            if name and name != "نامشخص":
                return name
    return None


def _backfill_anonymous(
    db: SessionDB,
    canonical_names: list[str],
    aliases: dict[str, str],
    apply: bool,
) -> int:
    assert db._conn is not None
    rows = db._conn.execute(
        "SELECT id, summary FROM sessions WHERE user_id IS NULL AND summary IS NOT NULL"
    ).fetchall()
    updated = 0
    for r in rows:
        guess = _guess_user_from_summary(r["summary"] or "")
        if not guess:
            continue
        canon = SessionDB._canonicalize_name(guess, canonical_names, aliases)
        if canon not in canonical_names:
            continue  # Only back-fill for known canonical users
        uid = db.get_or_create_user(
            canon, known_names=canonical_names, known_aliases=aliases
        )
        print(f"  session {r['id']:>4}  → {canon} (id={uid})   [heard: {guess!r}]")
        if apply:
            db._conn.execute(
                "UPDATE sessions SET user_id = ? WHERE id = ?", (uid, r["id"])
            )
        updated += 1
    if apply:
        db._conn.commit()
    return updated


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="Actually write changes (default: dry-run)")
    p.add_argument("--backfill-anonymous", action="store_true",
                   help="Also assign anonymous sessions to users when the summary reveals the name.")
    p.add_argument("--db", type=str, default=None, help="Path to sessions.db (defaults to config path)")
    args = p.parse_args()

    db_path = Path(args.db) if args.db else (Path(config.SESSION_DB_PATH) if config.SESSION_DB_PATH else Path("sessions.db"))
    if not db_path.exists():
        print(f"DB not found at {db_path}", file=sys.stderr)
        return 1
    print(f"Target DB: {db_path}")

    if args.apply:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = db_path.with_suffix(db_path.suffix + f".bak-{stamp}")
        shutil.copy2(db_path, backup)
        print(f"Backup written to: {backup}")

    db = SessionDB(db_path=db_path)
    try:
        canonical = config.ENGLISH_TEACHER_USER_NAMES or [
            u["name"] for u in db.list_users()
        ]
        aliases = dict(config.ENGLISH_TEACHER_USER_ALIASES or {})
        print(f"Canonical names: {canonical or '(none configured)'}")
        if aliases:
            print(f"Aliases: {aliases}")
        print()

        pairs = _dupes_to_merge(db, canonical, aliases)
        if not pairs:
            print("No duplicate users detected.")
        else:
            print(f"Would merge {len(pairs)} pair(s):")
            for src, tgt in pairs:
                print(f"  user id {src}  →  user id {tgt}")
            if args.apply:
                for src, tgt in pairs:
                    db.merge_users(src, tgt)
                print("Merges applied.")

        if args.backfill_anonymous:
            print("\nBack-filling anonymous sessions from summaries…")
            n = _backfill_anonymous(db, canonical, aliases, apply=args.apply)
            verb = "updated" if args.apply else "would update"
            print(f"{verb} {n} session(s).")

        if not args.apply:
            print("\nDry run — nothing was written. Re-run with --apply.")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
