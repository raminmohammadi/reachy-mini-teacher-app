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

A feature-rich conversational AI app for [Reachy Mini](https://pollen-robotics.com/reachy-mini/) that combines **Google Gemini Live** real-time audio with a **fully local AI pipeline** (Whisper + Ollama + Kokoro). Designed for Persian (Farsi) speaking users with a built-in live web dashboard.

---

## ✨ Features

### 🎙️ Dual AI Backends
| Mode | STT | LLM | TTS |
|------|-----|-----|-----|
| **Gemini Live** (default) | Gemini built-in | Gemini Flash Live | Gemini built-in |
| **Local** | Faster-Whisper | Ollama (any model) | Kokoro-ONNX |
| **OpenAI** | OpenAI built-in | GPT-4o Realtime | OpenAI built-in |

### 🖥️ Live Web Dashboard
When launched via the Reachy Mini Apps dashboard, a **FastAPI web UI** is automatically served at `http://localhost:7860` with:
- **Live transcript** — user and assistant turns rendered as they happen, auto-scrolling
- **Session stats** — mode, active profile, session ID, message count, uptime
- **Profile switcher** — switch between personas mid-session with one click
- **Online / offline** status badge that tracks the API connection

### 🧑‍🏫 Profiles
- **Default** — General-purpose Farsi-speaking AI companion with real-time conversation, head tracking, emotions, and weather lookup.
- **English Teacher** — Structured 7-unit curriculum for elderly Persian speakers learning English. Conversational method: scene-setting → modelling → role-play → gentle correction. Tracks progress across sessions via session summaries.

### 🛠️ Built-in Tools
| Tool | Description |
|------|-------------|
| `head_tracking` | Follows the user's face with the camera |
| `play_emotion` | Plays expressive robot animations |
| `dance` | Executes choreographed movement sequences |
| `move_head` | Points the head toward a specific direction |
| `check_weather` | Fetches real weather via Open-Meteo (no API key needed) — responds in Farsi |
| `camera` | Captures a snapshot and describes what it sees |
| `switch_persona` | Switches between available profiles mid-conversation |
| `do_nothing` | Explicit wait — stops the AI from filling silence |
| `stop_dance` / `stop_emotion` | Cancels ongoing movements |

### 💾 Session Memory
- SQLite database stores full transcripts and per-session AI-generated summaries.
- On reconnect, the AI receives the current session transcript and never repeats itself or re-asks the user's name.
- The English Teacher reads summaries from previous sessions to resume the exact lesson unit.

### 🔄 Connection Robustness
- The 100 Hz robot control loop detects hardware disconnection and drops to 1 Hz reconnect retries, preventing log flooding and high CPU usage during drops.
- The Gemini Live handler automatically reconnects and resumes using a session resumption token.

---

## 🚀 Quick Start

### Requirements
- Reachy Mini robot (connected via USB or Wi-Fi)
- Python ≥ 3.10
- A `GEMINI_API_KEY` (for Gemini Live mode)

### Installation
```bash
pip install reachy_mini_teacher_app
```

### Configuration
Copy `.env.example` to `.env` and fill in your API key:
```env
GEMINI_API_KEY=your_key_here
APP_MODE=gemini                             # or: local, openai
GEMINI_MODEL=models/gemini-3.1-flash-live-preview
REACHY_MINI_CUSTOM_PROFILE=english_teacher  # omit for default profile
```

### Run via CLI
```bash
reachy-mini-teacher-app
```

### Run via Reachy Mini Apps Dashboard
The app exposes `ReachyMiniTeacherApp` — the dashboard discovers and launches it automatically.
The web UI starts at **http://localhost:7860** as soon as the app is running.

### Publish to HuggingFace Spaces
```bash
pip install reachy-mini-app-assistant
hf auth login --token $HF_TOKEN --add-to-git-credential
reachy-mini-app-assistant publish
# When prompted, enter the path to this directory
```

---

## 📁 Project Structure

```
reachy_mini_teacher_app/          ← project root (HF flat layout)
├── index.html                  # HuggingFace Space landing page
├── style.css                   # Space landing page styles
├── pyproject.toml
├── README.md
└── reachy_mini_teacher_app/      ← Python package
    ├── main.py                 # Entry point, ReachyMiniTeacherApp, FastAPI server
    ├── gemini_handler.py       # Gemini Live WebSocket handler
    ├── local_handler.py        # Local pipeline (Whisper + Ollama + Kokoro)
    ├── openai_handler.py       # OpenAI Realtime handler
    ├── moves.py                # 100 Hz robot movement control loop
    ├── session_db.py           # SQLite session & transcript storage
    ├── session_summarizer.py   # AI-generated session summaries
    ├── static/                 # Web dashboard (served at localhost:7860)
    │   ├── index.html
    │   ├── main.js
    │   └── style.css
    ├── profiles/
    │   ├── default/            # Default Farsi assistant profile
    │   └── english_teacher/    # Structured English teaching profile
    └── tools/
        ├── check_weather.py    # Real-time weather via Open-Meteo
        ├── head_tracking.py    # Face-following with camera
        ├── play_emotion.py     # Robot emotion animations
        └── ...
```

---

## 🔑 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google AI API key (required for Gemini mode) |
| `APP_MODE` | `gemini` | `gemini` / `local` / `openai` |
| `GEMINI_MODEL` | `models/gemini-3.1-flash-live-preview` | Gemini Live model name |
| `REACHY_MINI_CUSTOM_PROFILE` | *(default profile)* | Profile name to load on startup (e.g. `english_teacher`) |
| `OPENAI_API_KEY` | — | OpenAI API key (required for OpenAI mode) |
| `OPENAI_MODEL` | `gpt-4o-realtime-preview` | OpenAI Realtime model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (local mode) |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model to use (local mode) |
| `WHISPER_MODEL` | `base` | Faster-Whisper model size |
| `SESSION_DB_PATH` | *(auto)* | Path to the SQLite session database |
| `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` | *(built-in)* | Path to a custom profiles directory |

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.
