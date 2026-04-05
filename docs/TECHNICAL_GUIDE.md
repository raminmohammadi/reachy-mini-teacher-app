# Reachy Mini Local App — Technical Debugging Guide

> **Audience:** Engineers debugging, extending, or tuning the Reachy Mini Local App.
> **Last updated:** 2026-04-02

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Audio Pipeline Deep Dive](#2-audio-pipeline-deep-dive)
3. [Vision Pipeline Deep Dive](#3-vision-pipeline-deep-dive)
4. [Movement System Deep Dive](#4-movement-system-deep-dive)
5. [Tool Execution & Background Tasks](#5-tool-execution--background-tasks)
6. [Debugging Recipes](#6-debugging-recipes)
7. [Common Failure Modes](#7-common-failure-modes)
8. [Environment & Configuration Reference](#8-environment--configuration-reference)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                       LocalStream (console.py)                   │
│  ┌──────────┐    ┌───────────────────────┐    ┌──────────────┐  │
│  │ _record   │───▶│   Handler (Gemini /   │───▶│  _play_loop  │  │
│  │  _loop    │    │   OpenAI / Local)     │    │              │  │
│  └──────────┘    └───────────────────────┘    └──────────────┘  │
│       ▲                    │  │                      │           │
│       │                    │  │                      ▼           │
│  robot.media          tools  head_wobbler      robot.media      │
│  .get_audio_sample()        │                  .push_audio()    │
│                             ▼                                    │
│              BackgroundToolManager                               │
│              ├── camera   ──▶ CameraWorker                      │
│              ├── dance    ──▶ MovementManager                   │
│              ├── emotion  ──▶ MovementManager                   │
│              └── ...                                             │
└──────────────────────────────────────────────────────────────────┘
```

### Threading Model

| Thread              | Owner                | Frequency | Purpose                                |
|---------------------|----------------------|-----------|----------------------------------------|
| `asyncio` event loop| `LocalStream`        | —         | Handler I/O, tool dispatch             |
| `head-wobbler`      | `HeadWobbler`        | ~20 Hz    | Audio → head sway offsets              |
| `camera-worker`     | `CameraWorker`       | ~25 Hz    | Frame capture + face tracking          |
| `movement-worker`   | `MovementManager`    | 100 Hz    | `set_target` control loop              |

### Sample Rate Map

| Stage                        | Gemini         | OpenAI         |
|------------------------------|---------------|----------------|
| Robot mic → Handler          | Resampled → **16 kHz** | Resampled → **24 kHz** |
| Handler → AI API (WebSocket) | 16 kHz PCM16  | 24 kHz PCM16 (base64) |
| AI API → Handler (output)    | **24 kHz** PCM16 | **24 kHz** PCM16 (base64) |
| Handler → HeadWobbler        | 24 kHz (b64)  | 24 kHz (b64)   |
| SwayRollRT internal          | Always **16 kHz** (linear resample) | Same |
| Handler → robot speaker      | Resampled → robot output SR | Same |

---

## 2. Audio Pipeline Deep Dive

### 2.1 Input Path (Mic → AI)

The mic frames arrive from `robot.media.get_audio_sample()` and are forwarded through `LocalStream._record_loop()`:

```python
# console.py — _record_loop (simplified)
input_sr = self._robot.media.get_input_audio_samplerate()
frame = self._robot.media.get_audio_sample()
await self.handler.receive((input_sr, frame))
```

Each handler's `_send_loop()` then resamples and encodes:

```python
# gemini_handler.py — resamples to 16 kHz
if input_sr != GEMINI_INPUT_SAMPLE_RATE:  # 16000
    num_samples = int(len(audio_frame) * GEMINI_INPUT_SAMPLE_RATE / input_sr)
    audio_frame = resample(audio_frame, num_samples)  # scipy.signal.resample
pcm_int16 = audio_to_int16(audio_frame)
await session.send_realtime_input(
    audio={"data": pcm_int16.tobytes(), "mime_type": "audio/pcm;rate=16000"}
)

# openai_handler.py — resamples to 24 kHz, base64-encodes
if input_sr != OPENAI_SAMPLE_RATE:  # 24000
    num_samples = int(len(audio_frame) * OPENAI_SAMPLE_RATE / input_sr)
    audio_frame = resample(audio_frame, num_samples)
pcm_int16 = audio_to_int16(audio_frame)
b64_audio = base64.b64encode(pcm_int16.tobytes()).decode("ascii")
await conn.send({"type": "input_audio_buffer.append", "audio": b64_audio})
```

**Debugging tip — verify mic data is arriving:**
```python
# Add to _record_loop temporarily:
import logging; logger = logging.getLogger(__name__)
if frames_received % 100 == 0:
    logger.info("Mic frames: %d, sr=%d, shape=%s, dtype=%s, max=%.4f",
                frames_received, input_sr, frame.shape, frame.dtype,
                float(np.abs(frame).max()))
```

### 2.2 Output Path (AI → Speaker)

Both handlers decode incoming audio and push it to `output_queue`:

```python
# gemini_handler.py — _recv_loop
audio_arr = np.frombuffer(response.data, dtype=np.int16).reshape(1, -1)
hw.feed(base64.b64encode(response.data).decode())  # → HeadWobbler
await self.output_queue.put((GEMINI_OUTPUT_SAMPLE_RATE, audio_arr))

# openai_handler.py — _recv_loop
audio_bytes = base64.b64decode(event.delta)
audio_arr = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
hw.feed(event.delta)  # already base64
await self.output_queue.put((OPENAI_SAMPLE_RATE, audio_arr))
```

`LocalStream._play_loop()` then resamples to the robot's output SR:

```python
# console.py — _play_loop (simplified)
output_sr = self._robot.media.get_output_audio_samplerate()
audio_frame = audio_to_float32(audio_data)
if input_sr != output_sr and output_sr > 0:
    num_samples = int(len(audio_frame) * output_sr / input_sr)
    audio_frame = resample(audio_frame, num_samples)
self._robot.media.push_audio_sample(audio_frame)
```

**Debugging tip — check if audio is being received from AI:**
```python
# Add to _recv_loop temporarily:
logger.info("Audio delta: %d bytes, %d samples",
            len(audio_bytes), len(audio_arr[0]))
```



### 2.3 HeadWobbler — Audio-to-Motion Synchronization

`HeadWobbler` runs in a dedicated thread and converts base64 PCM audio into 6-DOF head offsets. Key parameters:

| Parameter            | Value  | Effect                                              |
|----------------------|--------|-----------------------------------------------------|
| `SAMPLE_RATE`        | 24 kHz | Expected input rate (matches AI output)             |
| `MOVEMENT_LATENCY_S` | 0.2s  | Delay to sync movement with speaker output          |
| `HOP_MS`             | 50 ms  | Movement update interval (~20 Hz)                   |

**Generation tracking** prevents "zombie movements" from a previous utterance:

```python
# head_wobbler.py — reset() increments generation
def reset(self):
    with self._state_lock:
        self._generation += 1
        self._base_ts = None
        self._hops_done = 0
    # Drain stale chunks
    while True:
        try:
            self.audio_queue.get_nowait()
        except queue.Empty:
            break
```

**Timing logic in `working_loop()`:**
```python
# Target time for the Nth hop:
target = base_ts + MOVEMENT_LATENCY_S + hops_done * hop_dt

now = time.monotonic()
if now - target >= hop_dt:
    # We're lagging — drop frames to catch up
    lag_hops = int((now - target) / hop_dt)
    ...
elif target > now:
    # We're ahead — sleep until the right moment
    time.sleep(target - now)
```

**Debugging tip — log timing drift:**
```python
# Add inside working_loop, after computing target:
drift_ms = (now - target) * 1000
if abs(drift_ms) > 30:
    logger.warning("HeadWobbler drift: %.1f ms (hops_done=%d)", drift_ms, hops_done)
```

### 2.4 SwayRollRT — Procedural Motion Engine

The `SwayRollRT` class in `speech_tapper.py` converts audio energy into sinusoidal head movements. It runs at 16 kHz internally regardless of input rate.

**Pipeline per audio chunk:**
1. `_to_float32_mono()` — normalize to [-1, 1] mono float32
2. `_resample_linear()` — resample to 16 kHz if needed (fast linear interp)
3. Carry buffer stitching — ensures 50ms HOP boundaries
4. `_rms_dbfs()` — compute loudness in dBFS over 20ms window
5. Hysteresis VAD — dual-threshold (ON: -35 dB, OFF: -45 dB)
6. Attack/Release envelope smoothing
7. 6 independent sine oscillators modulated by loudness × envelope

**Key tunables (the "personality knobs"):**

| Tunable          | Default | What it changes                              |
|------------------|---------|----------------------------------------------|
| `SWAY_MASTER`    | 1.5     | Global movement intensity multiplier         |
| `VAD_DB_ON`      | -35 dB  | Volume threshold to start moving             |
| `VAD_DB_OFF`     | -45 dB  | Volume threshold to stop moving              |
| `SWAY_F_PITCH`   | 2.2 Hz  | Nodding speed                                |
| `SWAY_F_YAW`     | 0.6 Hz  | Shaking speed                                |
| `SWAY_A_PITCH_DEG`| 4.5°   | Maximum nod amplitude                        |
| `SWAY_A_YAW_DEG` | 7.5°    | Maximum shake amplitude                      |
| `LOUDNESS_GAMMA` | 0.9     | < 1.0 = responsive to quiet; > 1.0 = only loud |

**Debugging tip — inspect SwayRollRT output:**
```python
from reachy_mini_teacher_app.audio.speech_tapper import SwayRollRT
import numpy as np

rt = SwayRollRT()
# Generate 1s of synthetic speech-like audio
fake_audio = (np.random.randn(16000) * 0.1).astype(np.float32)
hops = rt.feed(fake_audio, sr=16000)
for i, h in enumerate(hops):
    print(f"Hop {i}: pitch={h['pitch_deg']:.2f}° yaw={h['yaw_deg']:.2f}° "
          f"roll={h['roll_deg']:.2f}° x={h['x_mm']:.1f}mm")
```

---

## 3. Vision Pipeline Deep Dive

### 3.1 CameraWorker — Frame Buffering (camera_worker.py)

The camera worker polls at ~25 Hz (`time.sleep(0.04)`) and stores the latest frame:

```python
frame = self.reachy_mini.media.get_frame()
with self.frame_lock:
    self.latest_frame = frame
```

**Face tracking flow:**
1. `head_tracker.get_head_position(frame)` → returns normalized `eye_center`
2. Convert to pixel coordinates → `reachy_mini.look_at_image()` → 4×4 pose
3. Scale down by 0.6× (FOV compensation)
4. Store as 6-DOF offsets: `[x, y, z, roll, pitch, yaw]`

**Face-lost interpolation:**
- After 2.0s with no face → begin 1.0s linear interpolation back to neutral
- Uses `linear_pose_interpolation(start_pose, neutral_pose, t)` where `t ∈ [0, 1]`

**Debugging tip — check if camera is delivering frames:**
```python
# Temporary addition to CameraWorker.working_loop():
frame_count = 0
# Inside the loop after getting frame:
if frame is not None:
    frame_count += 1
    if frame_count % 100 == 0:
        logger.info("Camera: %d frames, shape=%s, dtype=%s",
                     frame_count, frame.shape, frame.dtype)
```

### 3.2 Vision Fallback — API-based Image Description (camera.py)

Since Live/Realtime APIs cannot process raw binary image data in tool responses, the `camera` tool uses a **text description fallback**:

```
Camera Frame → JPEG encode → base64 → Vision API → text description → tool response
```

**Routing logic (`_describe_image()`):**
```python
async def _describe_image(b64_jpeg: str, question: str) -> str:
    mode = config.APP_MODE
    if mode == "openai" and config.OPENAI_API_KEY:
        return await _openai_describe_image(b64_jpeg, question)  # gpt-4o-mini
    elif config.GEMINI_API_KEY:
        return await _gemini_describe_image(b64_jpeg, question)  # gemini-2.5-flash
    elif config.OPENAI_API_KEY:
        return await _openai_describe_image(b64_jpeg, question)  # fallback
    return "No vision API key configured"
```

**Tool response sanitization** prevents WebSocket protocol violations (error `1007`):
```python
# In both handlers' _on_tool_complete():
sanitized = {
    k: v for k, v in notification.result.items()
    if not (isinstance(v, str) and len(v) > 10_000)  # Strip base64 blobs
}
```

**Debugging tip — test vision pipeline standalone:**
```python
import asyncio, cv2, base64
from reachy_mini_teacher_app.tools.camera import _describe_image

# Capture a test frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

_, buf = cv2.imencode('.jpg', frame)
b64 = base64.b64encode(buf.tobytes()).decode()

result = asyncio.run(_describe_image(b64, "What do you see?"))
print("Vision result:", result[:500])
```

---

## 4. Movement System Deep Dive

### 4.1 MovementManager (moves.py)

The 100 Hz control loop in `MovementManager` composes:
- **Primary moves** (mutually exclusive): breathing, emotions, dances, goto
- **Secondary moves** (additive offsets): speech sway (HeadWobbler) + face tracking (CameraWorker)

```
Final Pose = compose_world_offset(primary_head_pose, secondary_offset)
```

**Key concept — secondary offsets are in world frame:**
```python
# moves.py — secondary offset composition
secondary_head = create_head_pose(
    x=speech_x + track_x,
    y=speech_y + track_y,
    z=speech_z + track_z,
    roll=speech_roll + track_roll,
    pitch=speech_pitch + track_pitch,
    yaw=speech_yaw + track_yaw,
)
final_head = compose_world_offset(primary_head, secondary_head)
```

**Idle behavior:** After inactivity (`mark_activity()` not called), the system starts an infinite `BreathingMove` that smoothly interpolates from the current pose to a breathing pattern.

### 4.2 Listening Mode

When the AI is listening (user is speaking), antennas are "frozen" to avoid mechanical noise being picked up by the mic. On unfreeze, they blend back smoothly.

---

## 5. Tool Execution & Background Tasks

### 5.1 BackgroundToolManager

Tools execute asynchronously so the AI can continue speaking:

```
Handler receives tool_call → BackgroundToolManager.start_tool()
    → asyncio.create_task(_run_tool()) → dispatch_tool_call()
    → Tool completes → ToolNotification → _on_tool_complete callback
    → Send result back to AI session → AI generates response
```

**Lifecycle tasks:**
- `_listener`: Watches notification queue, calls handler callbacks
- `_cleanup`: Every 5 min, purges old completed tools (>1 hour) and cancels stuck tools (>24 hours)

**Debugging tip — list running tools:**
```python
# If you have access to the handler:
running = handler.tool_manager.get_running_tools()
for t in running:
    elapsed = time.monotonic() - t.started_at
    print(f"  {t.tool_name} (id={t.id}) running for {elapsed:.1f}s")
```

---

## 6. Debugging Recipes

### 6.1 Enable Verbose Logging

```bash
reachy-mini-local-app --mode gemini --debug
```

Or set programmatically:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Target specific modules:
logging.getLogger("reachy_mini_teacher_app.gemini_handler").setLevel(logging.DEBUG)
logging.getLogger("reachy_mini_teacher_app.audio.head_wobbler").setLevel(logging.DEBUG)
logging.getLogger("reachy_mini_teacher_app.tools").setLevel(logging.DEBUG)
```

### 6.2 Diagnose "No Audio" Issues

```python
# 1. Check robot audio pipeline
from reachy_mini import ReachyMini
robot = ReachyMini()
print("Input SR:", robot.media.get_input_audio_samplerate())   # Should be > 0
print("Output SR:", robot.media.get_output_audio_samplerate()) # Should be > 0

# 2. Check if mic frames arrive
robot.media.start_recording()
import time; time.sleep(1)
sample = robot.media.get_audio_sample()
print("Sample:", sample is not None, sample.shape if sample is not None else "N/A")

# 3. Check if output works
import numpy as np
tone = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
robot.media.start_playing()
robot.media.push_audio_sample(tone)
```

### 6.3 Diagnose "Robot Not Moving" Issues

```python
# 1. Test SwayRollRT in isolation
from reachy_mini_teacher_app.audio.speech_tapper import SwayRollRT
import numpy as np
rt = SwayRollRT()
audio = np.random.randn(16000).astype(np.float32) * 0.3  # loud-ish
hops = rt.feed(audio, sr=16000)
print(f"Got {len(hops)} hops, first pitch={hops[0]['pitch_deg']:.2f}°" if hops else "No hops!")
# Expected: ~20 hops for 1s of audio

# 2. Verify HeadWobbler thread is alive
print("Wobbler thread alive:", handler.deps.head_wobbler._thread.is_alive())
print("Wobbler generation:", handler.deps.head_wobbler._generation)
print("Wobbler queue size:", handler.deps.head_wobbler.audio_queue.qsize())
```

### 6.4 Diagnose WebSocket / Session Errors

**Gemini `1007 invalid argument`:**
- Cause: Tool response contains binary data (base64 images) > WebSocket frame limit
- Fix: The sanitizer strips values > 10 KB. Check `_on_tool_complete()` in the handler.
- Verify: `logger.info("Sending tool response: %d bytes", len(output_str))`

**OpenAI session drops:**
- Check `OPENAI_API_KEY` is valid
- Check model name: `gpt-4o-realtime-preview` (not `gpt-4o`)
- Both handlers auto-reconnect with a 2s delay

### 6.5 Diagnose Camera/Vision Issues

```python
# 1. Check camera frames directly
from reachy_mini import ReachyMini
robot = ReachyMini()
frame = robot.media.get_frame()
print("Frame:", frame is not None, frame.shape if frame is not None else "N/A")

# 2. Check CameraWorker buffer
worker = handler.deps.camera_worker
frame = worker.get_latest_frame()
print("Buffered frame:", frame is not None)

# 3. Test vision API directly
import asyncio
from reachy_mini_teacher_app.tools.camera import _gemini_describe_image
# (Use a real base64 JPEG string)
result = asyncio.run(_gemini_describe_image(b64_jpeg, "Describe this image"))
print("Vision:", result[:200])
```

---

## 7. Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Input sample rate is 0` | Robot audio not ready | Wait for WebRTC media pipeline; check `reachy-mini` daemon |
| `1007 invalid argument` (Gemini) | Tool response too large | Already handled by sanitizer; check custom tools |
| Robot doesn't move during speech | HeadWobbler thread dead or generation mismatch | Check `_thread.is_alive()`, check `_generation` counter |
| Camera tool returns "No frame" | CameraWorker not started or camera hardware issue | Check `--no-camera` flag; check `robot.media.get_frame()` |
| Vision returns "No vision API key" | Missing `GEMINI_API_KEY` or `OPENAI_API_KEY` | Set at least one in `.env` |
| Head snaps back after face tracking | `face_lost_delay` too short | Increase `CameraWorker.face_lost_delay` (default 2.0s) |
| Audio-visual desync | `MOVEMENT_LATENCY_S` miscalibrated | Tune in `head_wobbler.py` (default 0.2s) |
| Tool call never returns | Background tool stuck | Check `tool_manager.get_running_tools()`; auto-timeout at 24h |
| `GoAway` from Gemini | Server-side session limit (~15 min) | Handler auto-reconnects; check logs for frequency |

---

## 8. Environment & Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_MODE` | `gemini` | `gemini`, `openai`, or `local` |
| `GEMINI_API_KEY` | — | Google AI Studio API key |
| `GEMINI_MODEL` | `models/gemini-3.1-flash-live-preview` | Gemini Live model |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-realtime-preview` | OpenAI Realtime model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server for local mode |
| `OLLAMA_MODEL` | `llama3.2` | Local LLM model |
| `WHISPER_MODEL` | `base` | Faster-Whisper model size |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `KOKORO_VOICE` | `af_heart` | Kokoro TTS voice |
| `LOCAL_VISION_MODEL` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | Local vision model |
| `HF_TOKEN` | — | Hugging Face token for gated models |

### CLI Flags

```bash
reachy-mini-local-app \
  --mode gemini|openai|local \  # Backend selection
  --debug \                     # Enable DEBUG logging
  --no-camera \                 # Disable camera worker
  --local-vision \              # Use local SmolVLM2 for vision
  --head-tracker \              # Enable YOLO face tracking
  --gradio \                    # Open Gradio web UI
  --robot-name <name>           # Robot hostname/IP
```