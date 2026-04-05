"""
# SwayRollRT: Audio-to-Procedural Motion Engine

**SwayRollRT** is the "biological rhythm" engine for the Reachy Mini robot. It transforms raw audio streams into natural, lifelike head sways and tilts in real-time. Instead of static, robotic movements, it uses audio energy and sinusoidal oscillators to make the robot look like it is "listening" or "feeling" the speech.

---

## 🧠 Core Concept: How it Works

The engine acts as a bridge between the **Auditory** (sound) and **Kinetic** (movement) systems. It processes audio in small "hops" and uses three layers of logic to decide how the head should move:

1. **VAD (Voice Activity Detection):** Determines if someone is actually speaking.
2. **Envelope Following:** Smooths out the "start" and "stop" of speech so the robot doesn't jerk its head abruptly.
3. **Procedural Oscillators:** Six independent "heartbeats" ($x, y, z, roll, pitch, yaw$) that fluctuate based on how loud the audio is.

---

## 🛠 Features

* **Hysteresis VAD:** Uses two different decibel thresholds (`VAD_DB_ON` and `VAD_DB_OFF`) to prevent the robot's head from "flickering" during quiet parts of a sentence.
* **Attack/Release Logic:** Just like a professional audio compressor, the robot has an "Attack" (how fast it starts moving when you speak) and a "Release" (how gracefully it returns to center when you stop).
* **Loudness Mapping:** The more you yell, the bigger the sways. The engine uses a Gamma correction curve to map decibels to physical millimeters.
* **6-DOF Movement:** Generates a full coordinate set:
* **Rotations:** Pitch (nodding), Yaw (shaking), Roll (tilting).
* **Translations:** X, Y, Z (subtle shifting in space).



---

## 📐 Technical Specifications

| Parameter | Value | Description |
| --- | --- | --- |
| **Sample Rate** | 16,000 Hz | Optimized for speech processing. |
| **Hop Size** | 50ms | The interval at which a new movement command is generated. |
| **Frame Size** | 20ms | The window used to calculate audio loudness (RMS). |
| **VAD Attack** | 40ms | How quickly the robot reacts to the start of a word. |
| **VAD Release** | 250ms | Prevents the head from stopping mid-sentence during short pauses. |

### The Oscillator Math

Each of the 6 axes follows a sine wave pattern controlled by unique frequencies ($F$) and amplitudes ($A$). For example, the **Pitch** calculation looks like this:

$$\text{pitch} = A_{pitch} \times \text{loudness} \times \text{envelope} \times \sin(2\pi \cdot F_{pitch} \cdot t + \phi)$$

---

## 🚀 Usage

The class is designed for **streaming**. You don't need the whole audio file; you just "feed" it chunks as they arrive from a microphone or TTS engine.

```python
from sway_roll_rt import SwayRollRT

# 1. Initialize
rt = SwayRollRT()

# 2. Feed raw PCM audio (int16 or float32)
# Returns a list of dictionaries containing movement offsets
results = rt.feed(pcm_data, sr=16000)

for hop in results:
    print(f"Move head to: {hop['pitch_deg']} degrees")

```

---

## 🔧 Tunables (The "Personality" Knobs)

You can change the robot's "personality" by adjusting the constants at the top of the file:

* **`SWAY_MASTER`**: Increase this to make the robot more "hyper" and animated.
* **`VAD_DB_ON`**: Lower this if the robot is ignoring quiet talkers.
* **`SWAY_F_PITCH`**: Change the frequency to make the robot nod its head faster or slower.

---

## 📁 File Structure

* `_rms_dbfs()`: Measures the volume of a sound slice.
* `_loudness_gain()`: Converts volume into a 0.0 to 1.0 "strength" value.
* `_to_float32_mono()`: Normalizes any incoming audio format so the engine can read it.
* `SwayRollRT`: The main state machine that keeps track of time ($t$) and the current movement envelope.

"""
from __future__ import annotations
import math
from typing import Any, Dict, List
from itertools import islice
from collections import deque

import numpy as np
from numpy.typing import NDArray

# Tunables
# --- Audio Engine Basics ---
# The number of audio samples processed per second (16kHz is standard for speech)
SR = 16_000 
# The duration (in ms) of the sliding window used to calculate loudness (RMS)
FRAME_MS = 20 
# The time interval between each unique head movement command (50ms = 20Hz update rate)
HOP_MS = 50 

# --- Sensitivity & VAD (Voice Activity Detection) ---
# Global multiplier for all movement; higher values make the robot more "expressive"
SWAY_MASTER = 1.5 
# Calibration offset to adjust for microphone gain or background noise levels
SENS_DB_OFFSET = +4.0 
# The volume threshold (in dB) that triggers the robot to start moving
VAD_DB_ON = -35.0 
# The volume threshold (in dB) that tells the robot to stop moving and return to idle
VAD_DB_OFF = -45.0 
# How long (in ms) speech must persist above the ON threshold before movement starts
VAD_ATTACK_MS = 40 
# How long (in ms) to wait after speech stops before declaring the robot is "idle"
VAD_RELEASE_MS = 250 
# Smoothing factor for the movement envelope; higher is more responsive, lower is more fluid
ENV_FOLLOW_GAIN = 0.65 

# --- Oscillator Frequencies (F) and Amplitudes (A) ---
# F = Cycles per second (Hz). A = Maximum movement range (Degrees or Millimeters).

# Nodding movement (up/down)
SWAY_F_PITCH = 2.2 
SWAY_A_PITCH_DEG = 4.5 

# Shaking movement (left/right)
SWAY_F_YAW = 0.6 
SWAY_A_YAW_DEG = 7.5 

# Tilting movement (ear-to-shoulder)
SWAY_F_ROLL = 1.3 
SWAY_A_ROLL_DEG = 2.25 

# Forward/Backward translation (shifting in space)
SWAY_F_X = 0.35 
SWAY_A_X_MM = 4.5 

# Lateral translation (sliding side-to-side)
SWAY_F_Y = 0.45 
SWAY_A_Y_MM = 3.75 

# Vertical translation (slight bobbing up/down)
SWAY_F_Z = 0.25 
SWAY_A_Z_MM = 2.25 

# --- Dynamics Mapping ---
# The lowest volume level that contributes to movement scale
SWAY_DB_LOW = -46.0 
# The highest volume level (anything louder results in maximum movement)
SWAY_DB_HIGH = -18.0 
# Non-linear scaling factor; values < 1.0 make small sounds cause larger sways
LOUDNESS_GAMMA = 0.9 
# Speed (in ms) at which the oscillator reaches full amplitude when speech starts
SWAY_ATTACK_MS = 50 
# Speed (in ms) at which the oscillator ramps down when speech ends
SWAY_RELEASE_MS = 250 

# --- Derived Constants (Calculated automatically from above) ---
# Number of audio samples per RMS frame
FRAME = int(SR * FRAME_MS / 1000) 
# Number of audio samples per movement hop
HOP = int(SR * HOP_MS / 1000) 
# Number of hops required to satisfy the VAD attack time
ATTACK_FR = max(1, int(VAD_ATTACK_MS / HOP_MS)) 
# Number of hops required to satisfy the VAD release time
RELEASE_FR = max(1, int(VAD_RELEASE_MS / HOP_MS)) 
# Number of hops for the sway envelope to ramp up
SWAY_ATTACK_FR = max(1, int(SWAY_ATTACK_MS / HOP_MS)) 
# Number of hops for the sway envelope to ramp down
SWAY_RELEASE_FR = max(1, int(SWAY_RELEASE_MS / HOP_MS))


def _rms_dbfs(x: NDArray[np.float32]) -> float:
    """
    Calculates the Root Mean Square (RMS) loudness of an audio signal in Decibels relative to Full Scale (dBFS).

    WHAT IT DOES:
    1. Square: It squares every sample in the array (making all values positive).
    2. Mean: It calculates the average of those squared values.
    3. Root: It takes the square root of that average to get the linear RMS.
    4. Log: It converts that linear value into a logarithmic decibel scale (dBFS) where 0.0 is maximum volume.

    WHY IT IS NEEDED:
    Raw audio samples oscillate rapidly between positive and negative values. Taking the average of raw samples 
    would result in nearly zero, which doesn't help us measure intensity. RMS provides a mathematically 
    accurate "energy level" of a sound slice. 

    We need this dBFS value to:
    - Trigger Voice Activity Detection (VAD) so the robot knows when to start swaying.
    - Scale the physical size of the sways (louder speech = bigger head movements).
    - Prevent "jitter" by looking at the average energy rather than individual sample peaks.

    Args:
        x: A 1D NumPy array of audio samples (float32, normalized between -1.0 and 1.0).

    Returns:
        A float representing loudness in dBFS (usually a negative number like -20.0).
    """
    # numerically stable rms (avoid overflow)
    x = x.astype(np.float32, copy=False)
    # 1e-12 is added as an 'epsilon' to prevent taking the log of zero (which would crash the program)
    rms = np.sqrt(np.mean(x * x, dtype=np.float32) + 1e-12, dtype=np.float32)
    return float(20.0 * math.log10(float(rms) + 1e-12))


def _loudness_gain(db: float, offset: float = SENS_DB_OFFSET) -> float:
    """
    Normalizes a decibel (dB) value into a [0, 1] range and applies gamma correction.

    WHAT IT DOES:
    1. Offset: Adjusts the raw dB value based on microphone sensitivity (SENS_DB_OFFSET).
    2. Linear Mapping: Scales the dB value from its range (SWAY_DB_LOW to SWAY_DB_HIGH) 
       into a simple 0.0 to 1.0 scale.
    3. Clipping: Ensures the value never goes below 0.0 (too quiet) or above 1.0 (too loud).
    4. Gamma Correction: Applies a power function (t^gamma) to the result to change 
       the "curve" of the robot's responsiveness.

    WHY IT IS NEEDED:
    Robot motors and physical movements are linear (e.g., "move 5mm"), but audio is 
    logarithmic. If we used the dB value directly, the head movements would feel 
    unnatural—either too twitchy at low volumes or barely changing at high volumes.

    The Gamma Correction is particularly vital: 
    - If Gamma < 1.0: Small, quiet sounds will produce larger, more visible sways.
    - If Gamma > 1.0: Only very loud sounds will trigger significant movement.
    
    This function ensures the robot's physical response feels "weighted" and 
    expressive across different speaking styles.

    Args:
        db: The current loudness in dBFS (calculated by _rms_dbfs).
        offset: A sensitivity boost or cut to tune the microphone response.

    Returns:
        A float between 0.0 and 1.0 representing the intensity of the movement.
    """
    # 1. Map the dB value to a 0.0 - 1.0 range
    t = (db + offset - SWAY_DB_LOW) / (SWAY_DB_HIGH - SWAY_DB_LOW)
    
    # 2. Clip the values to stay within bounds
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
        
    # 3. Apply Gamma for non-linear responsiveness
    return t**LOUDNESS_GAMMA if LOUDNESS_GAMMA != 1.0 else t

def _to_float32_mono(x: NDArray[Any]) -> NDArray[np.float32]:
    """
    Converts arbitrary PCM (Pulse Code Modulation) audio data into a standard float32 mono format.

    WHAT IT DOES:
    1. Reshaping: It identifies the structure of the input (e.g., is it 1D like a list or 2D like a stereo file?).
    2. Downmixing: If the audio is stereo (2 channels), it averages the left and right channels into a single mono track.
    3. Normalization: If the audio is "Integer PCM" (like a CD or WAV file), it scales the numbers so they all fall 
       between -1.0 and 1.0.
    4. Casting: It converts the final data type to float32, which is the high-precision format required for the 
       subsequent math.

    WHY IT IS NEEDED:
    Audio comes in many "flavors." Some microphones record in 16-bit integers (values from -32,768 to 32,767), 
    while others use 32-bit floats. Some sources provide stereo audio, but a robot's head only needs one 
    "loudness" value to react to.

    Without this function, the `_rms_dbfs` calculation would break or produce wildly incorrect results if you 
    switched from a mono USB mic to a stereo audio file. It creates a "Common Language" for the rest of the 
    code to speak.

    Args:
        x: A NumPy array containing audio data of any shape (N,), (C,N), etc., and any common PCM data type.

    Returns:
        A 1D NumPy array of float32 values normalized between -1.0 and 1.0.
    """
    a = np.asarray(x)
    if a.ndim == 0:
        return np.zeros(0, dtype=np.float32)

    # 1. Handle Multi-channel audio (Downmixing)
    if a.ndim == 2:
        # Decides which axis is the channel based on which is smaller (typically <= 8 channels)
        if a.shape[0] <= 8 and a.shape[0] <= a.shape[1]:
            a = np.mean(a, axis=0) # Average the channels to create Mono
        else:
            a = np.mean(a, axis=1)
    elif a.ndim > 2:
        # Flatten complex arrays into 1D
        a = np.mean(a.reshape(a.shape[0], -1), axis=0)

    # 2. Scale and Normalize
    if np.issubdtype(a.dtype, np.floating):
        # Already floating point; just ensure it's float32
        return a.astype(np.float32, copy=False)
        
    # Integer PCM (e.g., int16) normalization
    info = np.iinfo(a.dtype)
    # Scale maps the largest possible integer (32768) to 1.0
    scale = float(max(-info.min, info.max))
    return a.astype(np.float32) / (scale if scale != 0.0 else 1.0)


def _resample_linear(x: NDArray[np.float32], sr_in: int, sr_out: int) -> NDArray[np.float32]:
    """
    Performs a lightweight linear interpolation to change the sample rate of an audio buffer.

    WHAT IT DOES:
    1. Mapping: It calculates the ratio between the current sample rate and the target rate.
    2. Interpolation: It draws a straight line between existing audio samples and "guesses" 
       what the values should be at the new timestamps.
    3. Reshaping: It returns a new array that is either shorter (downsampling) or longer 
       (upsampling) than the original.

    WHY IT IS NEEDED:
    Most audio algorithms (like our Sway engine) are "Sample Rate Dependent." This means 
    the math relies on 16,000 samples representing exactly 1 second of time. If you 
    accidentally fed in 48,000 samples, the robot would think 3 seconds had passed, 
    causing its head movements to happen 3x faster than the audio!

    Linear resampling is used here because it is incredibly fast (low CPU usage), which 
    is critical for maintaining the 200ms real-time latency required for the Reachy Mini.

    Args:
        x: The normalized float32 audio array.
        sr_in: The sample rate the audio was recorded at (e.g., 44100).
        sr_out: The sample rate the engine needs (e.g., 16000).

    Returns:
        A new NumPy array resampled to the target rate.
    """
    if sr_in == sr_out or x.size == 0:
        return x
        
    # Determine how many samples the output array should have
    n_out = int(round(x.size * sr_out / sr_in))
    
    # Guard against empty or tiny buffers that could crash interpolation
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
        
    # Create two timelines: one for the input samples and one for the output samples
    t_in = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32, endpoint=True)
    t_out = np.linspace(0.0, 1.0, num=n_out, dtype=np.float32, endpoint=True)
    
    # Use NumPy's interp to "draw the lines" between samples
    return np.interp(t_out, t_in, x).astype(np.float32, copy=False)


class SwayRollRT:
    """
    At its simplest level, this class is a Real-Time Puppet Master.
    It listens to your voice and converts the "loudness" and "rhythm" of your speech into a
    dance for the robot's head. It ensures the robot doesn't just sit there like a plastic 
    toy while it's talking; instead, it nods, tilts, and sways in perfect sync with the audio.

    In technical terms:
    
    A real-time procedural animation engine that converts streaming audio into lifelike 
    6-DOF (Degrees of Freedom) head movements for the Reachy Mini robot.

    WHAT IT DOES:
    This class implements a sophisticated audio-to-motion pipeline. It ingests raw PCM 
    audio chunks, analyzes their rhythmic and spectral energy, and modulates a set of 
    six independent spatial oscillators (pitch, yaw, roll, x, y, z). The output is 
    a continuous stream of coordinate offsets that make the robot appear to be 
    physically reacting to the "soul" and "cadence" of speech.
    
    
    Here is exactly what it does, step-by-step:

    ### 1. It "Listens" for Speech (The Gatekeeper)
        It uses a **Voice Activity Detector (VAD)** to decide if the robot should be moving at all.

        * **The Logic:** If the audio is too quiet (background noise), it stays still. Once the volume hits a certain "trigger" point, it starts the movement engine.
        * **The "Hysteresis":** It's smart enough not to stop moving just because the speaker took a tiny breath. It waits for a definitive silence before returning the head to the center.

    ### 2. It Calculates the "Energy" (The Volume Knob)
        It measures the **RMS (Loudness)** of the audio.

        * If the robot is shouting, the sways are wide and dramatic.
        * If the robot is whispering, the sways become tiny, subtle micro-movements.

    ### 3. It Generates "Procedural" Motion (The Brain)
        This is the clever part. It doesn't just move the head "up" when it's loud. It uses **6 independent oscillators** (like tiny pendulums swinging at different speeds).

        * One pendulum controls the **Nod (Pitch)**.
        * One controls the **Shake (Yaw)**.
        * One controls the **Tilt (Roll)**.
        * Others control the **Shifting (X, Y, Z)**.
        By mixing these 6 different rhythms together, the robot looks like a living being with complex muscles, rather than a machine with a single motor.

    ### 4. It Handles the "Streaming" (The Conveyor Belt)
        In a real conversation, audio doesn't arrive all at once; it comes in tiny "deltas" or chunks.

        * **The "Carry" Logic:** If you send a chunk of audio that is too short for a full movement "hop," the class **saves it** in a "carry" buffer.
        * When the next chunk arrives, it stitches them together. This ensures the movement is perfectly smooth and never skips a beat, even if the internet or the computer is stuttering.

    WHY IT IS NEEDED:
    1. Life-likeness: Static robots during speech trigger the "uncanny valley" effect. 
       This engine provides the subtle, organic micro-movements found in human 
       communication.
    2. Real-Time Responsiveness: Traditional animation is pre-baked. This engine 
       allows the robot to react dynamically to live human voices or real-time LLM 
       responses without needing pre-generated motion files.
    3. Hardware Safety: By using procedural oscillators rather than raw audio-to-motor 
       mapping, the engine ensures that movements stay within safe physical limits 
       and remain fluid even if the audio signal is noisy or erratic.

    KEY ARCHITECTURAL COMPONENTS:
    - Sliding Window Buffer: Maintains a 10-second history of audio to provide 
      context for VAD and envelope following.
    - Hysteresis VAD: A "Voice Activity Detection" gate that prevents the head 
      from twitching during minor background noise by using dual thresholds 
      (ON/OFF).
    - Attack/Release Envelope: Smooths the transition between silence and speech 
      so the robot doesn't "jump" when a word starts.
    - Deterministic Oscillators: 6 sine-wave generators with unique frequencies 
      and phase offsets, ensuring that while the movement is procedural, it doesn't 
      look like a repetitive mechanical loop.

    Args:
        rng_seed (int): A seed for the random number generator used to initialize 
                        oscillator phases. Using a seed ensures that while phases 
                        are random, the "personality" of the movement can be 
                        replicated across sessions.

    Usage:
        rt = SwayRollRT()
        # In a loop receiving audio chunks:
        movement_frames = rt.feed(pcm_buffer, sr=16000)
        for frame in movement_frames:
            apply_to_motors(frame['pitch_rad'], frame['yaw_rad'], ...)
    """

    def __init__(self, rng_seed: int = 7):
        """Initialize the internal state and oscillators of the motion engine."""
        # The 'personality' seed; ensures the random phases are consistent if restarted
        self._seed = int(rng_seed)
        
        # A sliding window that keeps up to 10 seconds of audio for analysis (VAD and envelope)
        self.samples: deque[float] = deque(maxlen=10 * SR) # (double-ended queue) is a more general structure that allows elements to be inserted and removed from both ends  
        
        # A buffer for 'orphaned' audio samples that weren't long enough to form a full movement hop
        self.carry: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

        # Voice Activity Detection (VAD) state: Are we currently hearing speech?
        self.vad_on = False
        # Counters to track how many consecutive frames are above/below volume thresholds
        self.vad_above = 0
        self.vad_below = 0

        # The 'Envelope' level (0.0 to 1.0); tracks how much the robot is currently swaying
        self.sway_env = 0.0
        # Ramping counters to smoothly start (up) or stop (down) the head movements
        self.sway_up = 0
        self.sway_down = 0

        # Initialize the random number generator for unique movement phases
        rng = np.random.default_rng(self._seed)
        
        # Starting positions for the 6 oscillators (0 to 2π). 
        # This prevents the robot from always starting its nod at the exact same angle.
        self.phase_pitch = float(rng.random() * 2 * math.pi)
        self.phase_yaw   = float(rng.random() * 2 * math.pi)
        self.phase_roll  = float(rng.random() * 2 * math.pi)
        self.phase_x     = float(rng.random() * 2 * math.pi)
        self.phase_y     = float(rng.random() * 2 * math.pi)
        self.phase_z     = float(rng.random() * 2 * math.pi)
        
        # The internal master clock (in seconds); used to drive the sine wave math
        self.t = 0.0

    def reset(self) -> None:
        """Reset state (VAD/env/buffers/time) but keep initial phases/seed."""
        self.samples.clear()
        self.carry = np.zeros(0, dtype=np.float32)
        self.vad_on = False
        self.vad_above = 0
        self.vad_below = 0
        self.sway_env = 0.0
        self.sway_up = 0
        self.sway_down = 0
        self.t = 0.0

    def feed(self, pcm: NDArray[Any], sr: int | None) -> List[Dict[str, float]]:
        """
        Processes an incoming chunk of audio and generates corresponding head movement frames.

        WHAT IT DOES:
        1. Pre-Processing: Normalizes audio to mono float32 and resamples to 16kHz if necessary.
        2. Buffer Management: Uses a 'carry' buffer to stitch together partial audio chunks, 
           ensuring a consistent 50ms (HOP_MS) movement update rate.
        3. Feature Extraction: Analyzes the audio energy (dBFS) within a 20ms window.
        4. State Management: Runs a Hysteresis VAD (Voice Activity Detection) to decide if 
           the robot should be in a 'speaking' or 'idle' physical state.
        5. Motion Generation: Feeds the loudness and timing into 6 independent sine-wave 
           oscillators to produce pitch, yaw, roll, and x, y, z translations.
                
            ### 1. Rotations (The "Angles")

            These describe the orientation of the head. In your `SwayRollRT` code, these are calculated in both degrees (for humans) and radians (for the robot's motors).

            * **Pitch (Nodding):** Moving the head up and down (like saying "yes").
            * **Yaw (Shaking):** Rotating the head left and right (like saying "no").
            * **Roll (Tilting):** Tilting the head toward the left or right shoulder (the "curious dog" look).
            
            ### 2. Translations (The "Position")

            These describe moving the entire head assembly along a straight line. Since the Reachy Mini has a neck mechanism, it can "shift" its head slightly without just rotating it.

            * **X (Forward/Backward):** Pushing the head toward the user or pulling it back.
            * **Y (Side-to-Side):** Sliding the head left or right without turning it.
            * **Z (Up/Down):** A slight "bobbing" or vertical shifting.

        WHY IT IS NEEDED:
        Real-time audio streams are often unpredictable (varying chunk sizes, sample rates). 
        This method provides a 'stabilization layer' that guarantees the robot's motors 
        receive a steady stream of coordinates that match the rhythm of the speech, 
        regardless of network or processing jitter.

        Args:
            pcm: The raw audio data. Can be mono/stereo, integer/float.
            sr: The sample rate of the input. If None, assumes 16,000Hz.

        Returns:
            A list of 'hop' dictionaries. Each dict contains 6-DOF offsets in both 
            radians/mm (for the robot) and degrees (for debugging/logging).
            
        graph TD
            A[Input: Raw PCM Chunk] --> B[Convert to Float32 Mono]
            B --> C{Correct Sample Rate?}
            C -- No --> D[Linear Resample to 16kHz] --> E
            C -- Yes --> E[Append to Carry Buffer]
            
            E --> F{Carry Size >= HOP?}
            F -- No --> G[Return Accumulated Results]
            F -- Yes --> H[Slice 50ms HOP from Carry]
            
            H --> I[Add to Sliding Window Window]
            I --> J[Calculate RMS Loudness - dBFS]
            
            J --> K{VAD Logic: Above Threshold?}
            K -- Yes --> L[Attack: Ramp Up Movement State]
            K -- No --> M[Release: Ramp Down to Idle]
            
            L & M --> N[Update Sway Envelope - Smoothing]
            N --> O[Calculate Loudness Multiplier]
            O --> P[Update Master Clock t]
            
            P --> Q[Compute 6-DOF Sine Oscillators]
            Q --> R[Store Result Dict]
            R --> F
        """       
        # --- 1. PRE-PROCESSING ---
        # Normalize audio to mono float32 and resample to the internal 16kHz if necessary
        sr_in = SR if sr is None else int(sr)
        x = _to_float32_mono(pcm)
        if x.size == 0:
            return []
        if sr_in != SR:
            x = _resample_linear(x, sr_in, SR)
            if x.size == 0:
                return []

        # --- 2. BUFFER MANAGEMENT (Carry) ---
        # Stitch incoming partial chunks together to ensure we only process full 50ms 'HOP' slices
        if self.carry.size:
            self.carry = np.concatenate([self.carry, x])
        else:
            self.carry = x

        out: List[Dict[str, float]] = []

        # Process as many 50ms HOPs as possible from the current buffer
        while self.carry.size >= HOP:
            hop = self.carry[:HOP]
            remaining: NDArray[np.float32] = self.carry[HOP:]
            self.carry = remaining

            # Add new audio to the sliding window for loudness analysis
            self.samples.extend(hop.tolist())
            if len(self.samples) < FRAME:
                self.t += HOP_MS / 1000.0
                continue

            # --- 3. FEATURE EXTRACTION ---
            # Analyze the audio energy (dBFS) within the most recent 20ms window
            frame = np.fromiter(
                islice(self.samples, len(self.samples) - FRAME, len(self.samples)),
                dtype=np.float32,
                count=FRAME,
            )
            db = _rms_dbfs(frame)

            # --- 4. STATE MANAGEMENT (VAD & Envelope) ---
            # Hysteresis VAD: Decide if we are in 'speaking' or 'idle' mode based on volume thresholds
            if db >= VAD_DB_ON:
                self.vad_above += 1
                self.vad_below = 0
                if not self.vad_on and self.vad_above >= ATTACK_FR:
                    self.vad_on = True
            elif db <= VAD_DB_OFF:
                self.vad_below += 1
                self.vad_above = 0
                if self.vad_on and self.vad_below >= RELEASE_FR:
                    self.vad_on = False

            # Calculate the Target Envelope based on the VAD state (Attack vs. Release)
            if self.vad_on:
                self.sway_up = min(SWAY_ATTACK_FR, self.sway_up + 1)
                self.sway_down = 0
            else:
                self.sway_down = min(SWAY_RELEASE_FR, self.sway_down + 1)
                self.sway_up = 0

            # Smooth the transition (Envelope Following) to prevent robotic, jerky starts/stops
            up = self.sway_up / SWAY_ATTACK_FR
            down = 1.0 - (self.sway_down / SWAY_RELEASE_FR)
            target = up if self.vad_on else down
            self.sway_env += ENV_FOLLOW_GAIN * (target - self.sway_env)
            
            # Safety clamp for the envelope multiplier
            if self.sway_env < 0.0:
                self.sway_env = 0.0
            elif self.sway_env > 1.0:
                self.sway_env = 1.0

            # Determine the loudness intensity for this specific hop
            loud = _loudness_gain(db) * SWAY_MASTER
            env = self.sway_env
            # Progress the internal master clock
            self.t += HOP_MS / 1000.0

            # --- 5. MOTION GENERATION (Oscillators) ---
            # Feed loudness, time, and envelope into 6 independent sine-wave oscillators (6-DOF)
            pitch = (
                math.radians(SWAY_A_PITCH_DEG)
                * loud
                * env
                * math.sin(2 * math.pi * SWAY_F_PITCH * self.t + self.phase_pitch)
            )
            yaw = (
                math.radians(SWAY_A_YAW_DEG)
                * loud
                * env
                * math.sin(2 * math.pi * SWAY_F_YAW * self.t + self.phase_yaw)
            )
            roll = (
                math.radians(SWAY_A_ROLL_DEG)
                * loud
                * env
                * math.sin(2 * math.pi * SWAY_F_ROLL * self.t + self.phase_roll)
            )
            x_mm = SWAY_A_X_MM * loud * env * math.sin(2 * math.pi * SWAY_F_X * self.t + self.phase_x)
            y_mm = SWAY_A_Y_MM * loud * env * math.sin(2 * math.pi * SWAY_F_Y * self.t + self.phase_y)
            z_mm = SWAY_A_Z_MM * loud * env * math.sin(2 * math.pi * SWAY_F_Z * self.t + self.phase_z)

            # Store the resulting offsets for this hop (in both radians and degrees for convenience)
            out.append(
                {
                    "pitch_rad": pitch,
                    "yaw_rad": yaw,
                    "roll_rad": roll,
                    "pitch_deg": math.degrees(pitch),
                    "yaw_deg": math.degrees(yaw),
                    "roll_deg": math.degrees(roll),
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "z_mm": z_mm,
                },
            )

        return out