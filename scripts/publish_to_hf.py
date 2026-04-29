#!/usr/bin/env python3
"""Publish the Reachy Mini Teacher App to a HuggingFace Space.

Called automatically by .github/workflows/publish.yml on every merge to main.
Uses the huggingface_hub Python API directly — no interactive prompts — so it
works reliably in CI without needing reachy-mini-app-assistant's interactive CLI.

Required environment variable
------------------------------
HF_TOKEN  — a HuggingFace User Access Token with Write permissions.
            Add it as a repository secret in:
            GitHub → Settings → Secrets and variables → Actions → New secret

Space visibility
----------------
Set SPACE_PRIVATE=false as a GitHub Actions variable (not secret) to make the
Space public. Default is private.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "reachy-mini-teacher-app"

# Files and directories that are never uploaded to the Space.
IGNORE_PATTERNS = [
    ".git/",
    ".github/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.egg-info/",
    ".eggs/",
    "cache/",
    "docs/",
    "scripts/",       # this file itself
    "tests/",
    "how_to_run.txt",
    ".env",           # never upload secrets
    ".env.*",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: Environment variable '{name}' is not set.", file=sys.stderr)
        print(
            "       For local use:  export HF_TOKEN=your_token\n"
            "       For CI:        add it as a GitHub repository secret.",
            file=sys.stderr,
        )
        sys.exit(1)
    return value


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    token = _require_env("HF_TOKEN")

    # SPACE_PRIVATE env var lets you override visibility without changing code.
    private = os.environ.get("SPACE_PRIVATE", "true").lower() != "false"

    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        print("ERROR: huggingface_hub is not installed.", file=sys.stderr)
        print("       Run: pip install --upgrade huggingface_hub", file=sys.stderr)
        sys.exit(1)

    # Resolve the HF username from the token — no hard-coding needed.
    try:
        user_info = whoami(token=token)
    except Exception as exc:
        print(f"ERROR: Could not authenticate with HuggingFace: {exc}", file=sys.stderr)
        print("       Check that HF_TOKEN is valid and has Write permissions.", file=sys.stderr)
        sys.exit(1)

    username = user_info["name"]
    repo_id = f"{username}/{PACKAGE_NAME}"
    visibility_label = "private" if private else "public"

    print(f"→ Authenticated as : {username}")
    print(f"→ Target Space     : https://huggingface.co/spaces/{repo_id}")
    print(f"→ Visibility       : {visibility_label}")

    api = HfApi(token=token)

    # Create the Space if it does not exist yet; silently update it if it does.
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        private=private,
        exist_ok=True,
    )
    print("  Space ready ✓")

    # Upload the full repository, skipping dev/CI-only paths.
    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    commit_msg = f"ci: publish from main ({sha})"

    api.upload_folder(
        folder_path=str(REPO_ROOT),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=commit_msg,
    )

    print(f"\n✅ Published successfully!")
    print(f"   https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
