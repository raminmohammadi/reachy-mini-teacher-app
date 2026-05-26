---
title: Reachy Mini Teacher App
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
tags:
  - reachy-mini
  - robotics
  - gemini
  - conversational-ai
  - english-teacher
  - farsi
  - multimodal
license: apache-2.0
---

# 🤖 Reachy Mini Teacher App

A conversational AI app for [Reachy Mini](https://pollen-robotics.com/reachy-mini/) that combines **Google Gemini Live** real-time audio with a **fully local AI pipeline** (Whisper + Ollama + Kokoro). Designed for Persian (Farsi) speaking users, with a built-in English teacher persona and a live web dashboard.

---

## ✨ Features

### 🎙️ Three AI Backends
| Mode | STT | LLM | TTS |
|------|-----|-----|-----|
| **Gemini Live** (default) | Gemini native audio | Gemini Flash Live | Gemini native audio |
| **Local** | Faster-Whisper | Ollama (any model) | Kokoro-ONNX |
| **OpenAI** | OpenAI built-in | GPT-4o Realtime | OpenAI built-in |

Gemini Live processes raw audio natively — no transcription step — which gives it far better understanding of accented and elderly speech.

### 🖥️ Live Web Dashboard
When launched, a **FastAPI web UI** is served at `http://localhost:7860` with:
- **Live transcript** — user and assistant turns rendered in real time, auto-scrolling
- **Session stats** — mode, active profile, session ID, message count, uptime
- **Profile switcher** — switch between personas mid-session with one click
- **Online / offline** status badge

### 🧑‍🏫 Profiles
- **Default** — General-purpose Farsi-speaking AI companion with real-time conversation, head tracking, emotions, and weather lookup.
- **English Teacher** — Structured 7-unit curriculum for elderly Persian speakers learning English. See [English Teacher](#-english-teacher-profile) section below.

### 👁️ Continuous Face Tracking
The robot follows the user's face automatically, all the time — no tool call needed. A background camera worker grabs frames from the Reachy Mini camera at 25 Hz and runs a YOLOv11n face detector at ~7 Hz; the detected face center is fed into the 100 Hz movement loop as a damped (0.6×) secondary offset on top of whatever primary move is playing. When no face is visible for 2 s the head smoothly glides back to neutral over 1 s.

Requires the `yolo_vision` extra (see [Installation](#installation)). On systems with a broken NVIDIA driver the app sets `CUDA_VISIBLE_DEVICES=""` so torch falls back to CPU cleanly — CPU is fast enough for the 7 Hz inference rate.

### 🛠️ Built-in Tools
| Tool | Description |
|------|-------------|
| `play_emotion` / `stop_emotion` | Plays / interrupts expressive robot animations |
| `dance` / `stop_dance` | Executes / interrupts choreographed movement sequences |
| `move_head` | Points the head in a specific direction |
| `check_weather` | Fetches real weather via Open-Meteo (no API key needed) |
| `camera` | Captures a snapshot and describes what it sees |
| `switch_persona` | Switches between available profiles mid-conversation |
| `remember_user_name` / `switch_user` | Persists / switches the active user across sessions |
| `set_user_level` | Saves the student's assessed English level (1–3) |
| `do_nothing` | Explicit wait — stops the AI from filling silence |

### 💾 Session Memory
- SQLite database (`sessions.db` at the project root) stores full transcripts and structured AI-generated summaries.
- Each summary has labelled fields: **کاربر** (user), **نتیجه** (pass/fail), **تمرین‌شده** (phrases practiced), **عملکرد** (performance), **تکرار** (what to repeat), **ادامه** (what to continue), **یادداشت** (teacher notes).
- Summaries are filtered by user — each user only ever sees their own history.
- The English Teacher reads the previous session's structured summary at session start and begins directly where it left off.

### 🔄 Connection Robustness
- Gemini Live sessions stay open for the full conversation. The `session.receive()` SDK limitation (which broke the loop after every AI turn) is worked around by using `session._receive()` directly.
- Genuine network disconnects trigger an automatic reconnect with the last 20 lines of transcript injected so the AI continues naturally.
- The 100 Hz robot control loop detects hardware disconnection and drops to 1 Hz retries, preventing log flooding and high CPU usage.

### 🎤 Voice Activity Detection (VAD) Tuning
All VAD settings use the correct snake_case field names required by the `google-genai` SDK (camelCase keys are silently ignored). The app always sets:
- `END_SENSITIVITY_LOW` — waits for a natural pause before declaring end-of-speech, preventing elderly/accented speakers from being cut off mid-sentence
- 1 200 ms silence window and 300 ms prefix padding (overridable via `.env`)

---

## 🧑‍🏫 English Teacher Profile

### Users
User names are configured in `.env` — there are no hard-coded names in the codebase:

```env
ENGLISH_TEACHER_USERS=Alice,Bob
```

Add as many comma-separated names as needed (one user, two users, or more). The AI listens to voice pitch at the start of each session, picks the most likely name from the list, confirms once, then calls `remember_user_name` silently. On subsequent sessions the name is already known and no question is asked. Leave the value blank to let the AI ask freely for any name.

### Student Level System
| Level | Label | Teaching style |
|-------|-------|---------------|
| 1 | مبتدی (Beginner) | Always translate, full guidance, patient pace |
| 2 | متوسط (Intermediate) | Occasional translation, some independent attempts |
| 3 | پیشرفته (Advanced) | No translation, natural conversation speed |

Level is assessed from the first few exchanges and persisted in the database via the `set_user_level` tool. It is re-assessed silently if the student's ability appears to have changed.

### Curriculum (7 units)
| Unit | Topic | Example phrases |
|------|-------|----------------|
| 1 | احوالپرسی پایه | Hello, Good morning, How are you? |
| 2 | معرفی خود | My name is…, Nice to meet you |
| 3 | اعداد و زمان | What time is it?, It's … o'clock |
| 4 | خانواده | This is my son/daughter, I have … children |
| 5 | خرید و رستوران | How much is this?, I want… |
| 6 | سلامتی و اورژانس | I feel sick, I need a doctor |
| 7 | آب‌وهوا | It's sunny/rainy, What's the weather like? |

Each day's unit advances only after the student passes (≥ 3 phrases used correctly and confidently). Incomplete days repeat the same unit.

### Teaching Method
Each session follows a **conversational arc** — not isolated drills:
1. **Session review** — one phrase from the previous session's `تکرار` (repeat) list is tested first.
2. **Scene-setting** — each new phrase is introduced inside a real-life scenario ("Imagine you walk into a doctor's office in the morning…").
3. **Two attempts max per phrase** — correct on attempt 1 → move on immediately; wrong → precise feedback + one more try → move on regardless.
4. **Full mini-conversation** — when all phrases are covered, the session ends with an unscripted roleplay that uses everything learned that day.

Honest feedback rule: **never say "آفرین" when pronunciation is wrong**. The AI must name exactly what was wrong (stress, consonant, speed, incomplete phrase) before asking again.

---

## 🚀 Quick Start

### Requirements
- Reachy Mini robot (connected via USB or Wi-Fi)
- Python ≥ 3.10
- A `GEMINI_API_KEY` (for Gemini Live mode)

### Installation
For developers (editable install from a checkout):
```bash
git clone <repo-url> reachy-mini-teacher-app
cd reachy-mini-teacher-app
python3 -m venv venv && source venv/bin/activate
pip install -e '.[yolo_vision]'
```

The `[yolo_vision]` extra pulls in `ultralytics`, `supervision`, and `torch` (CPU build) so continuous face tracking works out of the box. Without it the app still runs, but the camera worker logs a warning and the head stays still.

Other optional extras:
- `[local_vision]` — local SmolVLM2 vision model for the `camera` tool (no Gemini round-trip).
- `[mediapipe_vision]` — alternative head tracker via MediaPipe (face mesh).

### Configuration
Copy `.env.example` to `.env` and fill in your API key:
```env
GEMINI_API_KEY=your_key_here
APP_MODE=gemini                             # or: local, openai
GEMINI_MODEL=gemini-3.1-flash-live-preview
REACHY_MINI_CUSTOM_PROFILE=english_teacher  # omit for default profile
```

### Run
```bash
reachy-mini-teacher-app
```

The web dashboard starts at **http://localhost:7860** automatically. CLI flags:
- `--no-camera` — disable the camera worker entirely (no face tracking, no `camera` tool)
- `--head-tracker {yolo,mediapipe,None}` — choose tracker backend (default: `yolo`)
- `--local-vision` — use local SmolVLM2 instead of Gemini for the `camera` tool
- `--mode {gemini,openai,local}` — pick the AI backend (overrides `APP_MODE`)

### Desktop Launcher (Linux / GNOME)
For non-technical end users, install one-click desktop icons that handle daemon cycling, browser open, and live log streaming:
```bash
./scripts/install_launcher.sh
```
This installs two `.desktop` entries:
- **Reachy Teacher** — wakes the robot, starts the app, opens the dashboard.
- **Stop Reachy Teacher** — gracefully stops the app and puts the robot to sleep.

The launcher waits up to 60 s for the FastAPI dashboard to come up (torch + YOLO load is ~11 s on CPU), then opens the browser. Logs are written to `~/.local/share/reachy-mini-teacher-app/{app,launcher}.log`.

### Publish to HuggingFace Spaces
```bash
pip install reachy-mini-app-assistant
hf auth login --token $HF_TOKEN --add-to-git-credential
reachy-mini-app-assistant publish
```

---

## 📁 Project Structure

```
reachy-mini-teacher-app/          ← repo root
├── sessions.db                 # SQLite session database (git-ignored)
├── pyproject.toml
├── README.md
├── scripts/                    # Desktop launcher + install helpers
│   ├── install_launcher.sh
│   ├── launch_teacher_app.sh   # Foreground launcher with daemon cycling
│   ├── stop_teacher_app.sh
│   └── reachy-mini-teacher-app{,-stop}.desktop
└── reachy_mini_teacher_app/      ← Python package
    ├── main.py                 # Entry point, ReachyMiniTeacherApp, FastAPI server
    ├── gemini_handler.py       # Gemini Live handler (native audio, VAD, multi-turn)
    ├── local_handler.py        # Local pipeline (Whisper + Ollama + Kokoro)
    ├── openai_handler.py       # OpenAI Realtime handler
    ├── config.py               # Environment variable config (pydantic-settings)
    ├── moves.py                # 100 Hz robot movement control loop
    ├── camera_worker.py        # 25 Hz frame buffer + 7 Hz face inference thread
    ├── session_db.py           # SQLite session storage, curriculum, recap
    ├── session_summarizer.py   # Structured AI-generated session summaries
    ├── prompts.py              # Prompt loading and placeholder injection
    ├── static/                 # Web dashboard (served at localhost:7860)
    ├── vision/
    │   └── yolo_head_tracker.py  # YOLOv11n face detector (default tracker)
    ├── profiles/
    │   ├── default/            # Default Farsi assistant profile
    │   └── english_teacher/    # English teaching profile + instructions
    └── tools/
        ├── camera.py
        ├── check_weather.py
        ├── play_emotion.py
        ├── remember_user_name.py
        ├── set_user_level.py
        └── ...
```

---

## 🔑 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google AI API key (required for Gemini mode) |
| `APP_MODE` | `gemini` | `gemini` / `local` / `openai` |
| `GEMINI_MODEL` | `gemini-3.1-flash-live-preview` | Gemini Live model name |
| `REACHY_MINI_CUSTOM_PROFILE` | *(default profile)* | Profile to load on startup (e.g. `english_teacher`) |
| `ENGLISH_TEACHER_USERS` | *(empty — ask freely)* | Comma-separated user names for the English Teacher profile (e.g. `Alice,Bob`) |
| `VAD_SILENCE_DURATION_MS` | `1200` | Silence window before end-of-speech (ms) |
| `VAD_PREFIX_PADDING_MS` | `300` | Audio prefix captured before speech starts (ms) |
| `OPENAI_API_KEY` | — | OpenAI API key (required for OpenAI mode) |
| `OPENAI_MODEL` | `gpt-4o-realtime-preview` | OpenAI Realtime model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (local mode) |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model (local mode) |
| `WHISPER_MODEL` | `base` | Faster-Whisper model size |
| `SESSION_DB_PATH` | `sessions.db` | Path to the SQLite session database |
| `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` | *(built-in)* | Path to a custom profiles directory |

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.
