# Reachy Mini Local App

A multi-backend conversational AI application for the [Reachy Mini](https://www.pollen-robotics.com/) robot. Supports **Gemini Live**, **OpenAI Realtime**, and a **fully local** (Whisper + Ollama + Kokoro) AI pipeline — switchable via a single environment variable or CLI flag.

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install the package
pip install -e .

# 3. Configure your API keys (copy and edit)
cp src/reachy_mini_teacher_app/.env.example .env

# 4. Run (default: Gemini Live mode)
reachy-mini-local-app
```

## Switching Between AI Backends

### Via environment variable (`.env` or shell)

```bash
# Gemini Live (default)
APP_MODE=gemini
GEMINI_API_KEY=your-gemini-key

# OpenAI Realtime
APP_MODE=openai
OPENAI_API_KEY=sk-your-openai-key

# Fully local (no API keys needed)
APP_MODE=local
```

### Via CLI flag

```bash
reachy-mini-local-app --mode gemini
reachy-mini-local-app --mode openai
reachy-mini-local-app --mode local
```

The `--mode` flag takes precedence over the `APP_MODE` environment variable.

## Architecture

```
reachy_mini_teacher_app/
├── pyproject.toml               # Dependencies & project metadata
├── .env.example                 # All config variables documented
├── tests/
│   └── test_openai_integration.py  # Unit tests for OpenAI handler, mode switching, etc.
└── src/reachy_mini_teacher_app/
    ├── config.py                # Multi-mode config (Gemini / OpenAI / Local)
    ├── main.py                  # CLI entry point & Reachy Mini Apps plugin
    ├── console.py               # LocalStream: drives mic/speaker for any handler
    ├── gemini_handler.py        # GeminiLiveHandler (google-genai Live API)
    ├── openai_handler.py        # OpenAIRealtimeHandler (OpenAI Realtime WebSocket API)
    ├── local_handler.py         # LocalPipelineHandler (Whisper → Ollama → Kokoro)
    ├── moves.py                 # 100 Hz movement control loop
    ├── dance_emotion_moves.py   # Dance/emotion move types
    ├── camera_worker.py         # Camera frame buffer + head tracking
    ├── prompts.py               # System prompt builder
    ├── utils.py                 # Arg parsing, logging, vision setup
    ├── audio/                   # HeadWobbler + SwayRollRT (audio-reactive motion)
    ├── tools/                   # Full tool registry (camera, dance, emotions, head…)
    ├── vision/                  # SmolVLM2 local vision + YOLO tracker
    └── profiles/default/        # Default instructions + tools.txt
```

## AI Backends

### Gemini Live (`--mode gemini`)

Uses the Google Gemini Multimodal Live API for real-time bidirectional audio conversation.

- **Audio**: 16 kHz PCM16 input, 24 kHz PCM16 output
- **Features**: Server-side VAD, function calling, live audio streaming
- **Vision fallback**: When no local vision model is loaded, camera tool queries are routed to `gemini-2.5-flash` via the standard Gemini API
- **Config**: `GEMINI_API_KEY`, `GEMINI_MODEL`

### OpenAI Realtime (`--mode openai`)

Uses the OpenAI Realtime API (WebSocket-based) for low-latency multimodal interaction.

- **Audio**: 24 kHz PCM16 input and output
- **Features**: Server-side VAD, function calling, input audio transcription, auto-reconnect
- **Vision fallback**: Camera tool queries are routed to `gpt-4o-mini` via the standard OpenAI Chat Completions API
- **Config**: `OPENAI_API_KEY`, `OPENAI_MODEL` (default: `gpt-4o-realtime-preview`)

### Local Pipeline (`--mode local`)

Fully offline pipeline using open-source models — no API keys required.

- **STT**: Faster-Whisper (configurable model size and compute type)
- **LLM**: Ollama (any model, default: `llama3.2`)
- **TTS**: Kokoro-ONNX (configurable voice and language)
- **Config**: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `WHISPER_MODEL`, `KOKORO_VOICE`, etc.

## Configuration Reference

All configuration is done via environment variables (or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `APP_MODE` | `gemini` | AI backend: `gemini`, `openai`, or `local` |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GEMINI_MODEL` | `models/gemini-3.1-flash-live-preview` | Gemini model name |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-realtime-preview` | OpenAI Realtime model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `WHISPER_MODEL` | `base` | Faster-Whisper model size |
| `WHISPER_COMPUTE_TYPE` | `int8` | Whisper compute type |
| `WHISPER_DEVICE` | `cpu` | Whisper device (`cpu`/`cuda`) |
| `KOKORO_VOICE` | `af_heart` | Kokoro TTS voice |
| `KOKORO_LANG` | `en-us` | Kokoro TTS language |
| `HF_HOME` | `./cache` | HuggingFace cache directory |
| `LOCAL_VISION_MODEL` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | Local vision model |

## CLI Options

```
reachy-mini-local-app [OPTIONS]

Options:
  --mode {gemini,openai,local}  AI backend (overrides APP_MODE env var)
  --head-tracker {yolo,mediapipe}  Head tracking method
  --no-camera                   Disable camera
  --local-vision                Use local SmolVLM2 vision model
  --gradio                      Open Gradio web interface
  --debug                       Enable debug logging
  --robot-name NAME             Robot name for Zenoh topics
```

## Tools

All AI backends share the same tool system. Available tools:

| Tool | Description |
|---|---|
| `camera` | Take a photo and ask a question about it |
| `dance` | Play a named or random dance move |
| `stop_dance` | Stop current dance |
| `play_emotion` | Play a pre-recorded emotion |
| `stop_emotion` | Stop current emotion |
| `head_tracking` | Toggle head tracking |
| `move_head` | Move head in a direction |
| `do_nothing` | Stay still and silent |
| `task_status` | Check background task status |
| `task_cancel` | Cancel a background task |

Tools run asynchronously via `BackgroundToolManager`. Tool results are sanitized (large base64 strings stripped) before being sent back to the AI backend to prevent session crashes.

## Camera & Vision

The camera tool supports three vision processing modes:

1. **Local vision** (`--local-vision`): Uses SmolVLM2 for on-device image understanding
2. **API vision fallback**: When no local vision is loaded, images are sent to the current backend's vision API:
   - Gemini mode → `gemini-2.5-flash`
   - OpenAI mode → `gpt-4o-mini`
3. **Cross-backend fallback**: If the preferred API key is missing, it falls back to whichever API key is available

## Testing

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

## Optional Dependencies

```bash
# Local vision (SmolVLM2)
pip install -e ".[local_vision]"

# YOLO head tracking
pip install -e ".[yolo_vision]"

# MediaPipe head tracking
pip install -e ".[mediapipe_vision]"
```
