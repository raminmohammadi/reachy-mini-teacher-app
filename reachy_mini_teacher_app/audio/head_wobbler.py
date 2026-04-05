"""Moves head given audio samples."""

import time
import queue
import base64
import logging
import threading
from typing import Tuple
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from reachy_mini_teacher_app.audio.speech_tapper import HOP_MS, SwayRollRT


SAMPLE_RATE = 24000
MOVEMENT_LATENCY_S = 0.2  # seconds between audio and robot movement
logger = logging.getLogger(__name__)


class HeadWobbler:
    """
    This code defines the **`HeadWobbler`**, a synchronization utility designed for the Reachy Mini robot. 
    Its primary purpose is to create **procedural head animation** (subtle sways and tilts) that reacts to 
    audio in real-time, making the robot appear more lifelike during speech.

    # HeadWobbler: Audio-Reactive Robot Animation

    The `HeadWobbler` class bridges the gap between raw audio data and robotic motor control. It processes 
    incoming audio "deltas," calculates physical movement offsets based on the audio's rhythm/amplitude 
    (via the `SwayRollRT` algorithm), and ensures these movements happen in sync with the sound the robot is producing.

    ## Why is this needed?

    1. **Life-likeness (Uncanny Valley Avoidance):** Static robots look like "statues" when they speak. By adding subtle,
        audio-driven head sways, the robot appears to be "thinking" or "feeling" the speech rhythm.
    2. **Audio-Visual Synchronization:** Sound processing takes time, and motor commands have physical latency. 
        This class manages a **playback timeline** to ensure the head moves exactly when the audio is heard by a human observer.
    3. **Concurrency Management:** Audio arrives in chunks (deltas) from a network or generator. This class handles those 
        chunks in a background thread so the main application doesn't freeze while waiting for motor updates.

    ---

    ## How It Works (The Logic)

    ### 1. The Processing Pipeline

    1. **`feed(delta_b64)`**: Accepts base64 encoded PCM audio. It decodes the audio and pushes it into a thread-safe queue.
    2. **`working_loop()`**: A background thread that constantly monitors the queue.
    * It sends audio to a "Sway" algorithm (`SwayRollRT`) which returns a list of **Hops** (individual movement frames).
    * Each hop contains translation ($x, y, z$) and rotation (roll, pitch, yaw) values.



    ### 2. The Timing Mechanism

    To prevent the robot from moving too early or falling behind, the wobbler uses a **Latency Buffer**:

    * **`MOVEMENT_LATENCY_S` (0.2s)**: It intentionally waits a fraction of a second to allow the audio system to actually output the sound through the speakers.
    * **Lag Compensation**: If the system gets bogged down and the "target time" for a movement has already passed, the wobbler will **skip (drop)** movement frames to catch up to the real-time audio.

    ### 3. Generation Management

    Since the robot might stop speaking and start a new sentence abruptly, the class uses a **`_generation`** counter:

    * When `reset()` is called, the generation increments.
    * The background thread ignores any "old" audio chunks still sitting in the queue from a previous generation. This prevents the robot from "wobbling" to the ghost of a finished sentence.

    ---

    ## Key Components

    | Component | Responsibility |
    | --- | --- |
    | **`audio_queue`** | Stores incoming audio chunks to be processed. |
    | **`SwayRollRT`** | The "brain" that converts PCM audio waveforms into spatial coordinates. |
    | **`_apply_offsets`** | A callback function that sends the calculated $x, y, z, r, p, y$ to the robot's hardware. |
    | **`_state_lock`** | Ensures thread-safety when updating timestamps and generation IDs. |

    ---

    ## Usage Example

    ```python
    # 1. Define how to move the robot
    def move_robot_motors(offsets):
        x, y, z, roll, pitch, yaw = offsets
        # Send to Reachy's SDK here...

    # 2. Initialize and start
    wobbler = HeadWobbler(set_speech_offsets=move_robot_motors)
    wobbler.start()

    # 3. Feed audio data (usually from a TTS stream)
    wobbler.feed(audio_b64_string)

    # 4. Stop when done
    wobbler.stop()

    ```

    """

    def __init__(self, set_speech_offsets: Callable[[Tuple[float, float, float, float, float, float]], None]) -> None:
        """Initialize the head wobbler."""
        self._apply_offsets = set_speech_offsets
        self._base_ts: float | None = None
        self._hops_done: int = 0

        self.audio_queue: "queue.Queue[Tuple[int, int, NDArray[np.int16]]]" = queue.Queue() # class is inherently thread-safe.
        self.sway = SwayRollRT()

        # Synchronization primitives
        self._state_lock = threading.Lock()
        self._sway_lock = threading.Lock()
        self._generation = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def feed(self, delta_b64: str) -> None:
        """
        Thread-safe: Accepts base64 encoded PCM audio. It decodes the audio and pushes it into a thread-safe queue
        
        The variable `self._generation` is a "shared resource." The background thread needs to know it to decide if it should skip old audio, and the `feed` thread needs to read it to label new audio.
        * **The Lock:** `self._state_lock` acts like a "talking stick" or a bathroom key.
        * **The Action:** When the code says `with self._state_lock:`, it "grabs the key." If the other thread currently has the key, this thread will **stop and wait** until the key is returned.
        * **The Protection:** This ensures that `generation = self._generation` happens atomically—meaning no other thread can change the value of `_generation` while we are in the middle of reading it.

        """
        buf = np.frombuffer(base64.b64decode(delta_b64), dtype=np.int16).reshape(1, -1)
        
        with self._state_lock:
            generation = self._generation
        self.audio_queue.put((generation, SAMPLE_RATE, buf))

    def start(self) -> None:
        """Start the head wobbler loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Head wobbler started")

    def stop(self) -> None:
        """Stop the head wobbler loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.debug("Head wobbler stopped")

    def working_loop(self) -> None:
        """ A background thread that constantly monitors the queue. 
            Convert audio deltas into head movement offsets.
            
            graph TD
            
            A([Start working_loop]) --> B{Queue Empty?}
            B -- Yes --> C[Sleep MOVEMENT_LATENCY_S] --> B
            B -- No --> D[Get Chunk: generation, sr, data]
            
            D --> E{Generation Match?}
            E -- No --> F[Discard Chunk] --> B
            E -- Yes --> G{Base Timestamp set?}
            
            G -- No --> H[Set _base_ts = now] --> I
            G -- Yes --> I[Feed audio to Sway Algorithm]
            
            I --> J[Get List of 'Hops' Results]
            J --> K{Next Hop in Results?}
            
            K -- Yes --> L{Gen Change?}
            L -- Yes --> B
            L -- No --> M[Calculate Target Time]
            
            M --> N{Status?}
            N -- "Lagging (Too Late)" --> O[Calculate Frames to Drop]
            O --> P[Update Counter & Skip Hops] --> K
            
            N -- "Ahead (Too Early)" --> Q[Sleep until Target Time] --> R
            N -- "On Time" --> R[Convert to 6-DOF Offsets]
            
            R --> S[apply_offsets Callback]
            S --> T[Increment _hops_done] --> K
            
            K -- No --> U[Mark Queue task_done] --> B
        """
        # Think of hop_dt as the frame rate of the robot’s movement. Just as a movie might play at 24 frames per second, the robot's head movement is broken down into small slices called "hops."
        hop_dt = HOP_MS / 1000.0 # Duration of one Step in the process - All predefined - If audio is 1 seconds it means 50 hobs each for 20 ms

        logger.debug("Head wobbler thread started")
        while not self._stop_event.is_set():
            queue_ref = self.audio_queue
            try:
                # Blocks until an item is available or timeout occurs, avoiding artificial polling latency
                chunk_generation, sr, chunk = queue_ref.get(timeout=0.1)  # (gen, sr, data)
            except queue.Empty:
                continue

            try:
                with self._state_lock:
                    current_generation = self._generation
                if chunk_generation != current_generation:
                    """
                    The Reason: To prevent "Zombie Movements."
                        In a conversation, the user might interrupt the robot. When you call reset(), you increment the generation.
                        This step ensures that any audio already "in the pipe" (queued up) from the previous, interrupted sentence 
                        is immediately discarded so the robot doesn't keep wobbling to a sentence it's no longer speaking.
                    """
                    continue

                if self._base_ts is None:
                    with self._state_lock:
                        if self._base_ts is None:
                            """
                            The Reason: Establishing a "Zero Hour."
                                To stay in sync with audio, you need a shared reference point. By marking the exact moment the first audio chunk is processed,
                                the robot creates a timeline. Every subsequent movement is calculated as an offset from this "Zero Hour," ensuring that the 50th
                                "hop" of movement happens exactly $50 \times 20ms$ after the start.
                            """
                            self._base_ts = time.monotonic() # Monotonic Clock - Can not go Backward
                            

                pcm = np.asarray(chunk).squeeze(0)
                with self._sway_lock:
                    results = self.sway.feed(pcm, sr)

                i = 0
                while i < len(results):
                    with self._state_lock:
                        if self._generation != current_generation:
                            break
                        base_ts = self._base_ts
                        hops_done = self._hops_done

                    if base_ts is None:
                        base_ts = time.monotonic()
                        with self._state_lock:
                            if self._base_ts is None:
                                self._base_ts = base_ts
                                hops_done = self._hops_done
                    """
                    The Step: target = base_ts + MOVEMENT_LATENCY_S + ...
                        The Reason: Account for System Overhead.
                        There is a delay between sending an audio buffer to the speakers and the sound actually coming out.
                        By adding a 200ms buffer, you give the system enough "breathing room" to process the math and send 
                        the motor commands so that the physical movement arrives at the robot's neck at the exact same moment the sound waves hit the air.
                                                
                        target is an absolute point in time on the system's clock. It answers the question: "At exactly what millisecond should this specific head-tilt happen?"
                        
                        Imagine the robot is about to say "Hello."The Start: The audio starts at time.monotonic() = 100.00s. This is your base_ts.The Goal: We want to process 
                        the 10th frame of movement. Each frame (hop_dt) is 0.02s.The Latency: We know the speakers take 0.20s to actually make sound.The Calculation:
                            Target = 100.00 + 0.20 + (10 \times 0.02) = 100.40s
                        
                        The code now knows that the 10th head movement must happen exactly at 100.40 seconds on the system clock.
                            If the current time is 100.35s, the code sees it's too early and tells the thread to sleep for 0.05s.
                            If the current time is 100.45s, the code realizes it's late and drops (skips) that frame to catch up.
                    """
                    target = base_ts + MOVEMENT_LATENCY_S + hops_done * hop_dt
                    now = time.monotonic()

                    if now - target >= hop_dt:
                        """
                        The Reason: Prioritizing Real-Time over Completeness.
                        If the computer slows down for a split second, the robot falls behind. If you simply played every frame, the head would be "lagging"
                        behind the voice (like a badly dubbed movie). This step calculates exactly how many frames the robot is behind and skips them. 
                        It sacrifices "smoothness" for a moment to ensure the head position "snaps" back to being perfectly in sync with the current audio.
                        """
                        lag_hops = int((now - target) / hop_dt)
                        drop = min(lag_hops, len(results) - i - 1)
                        if drop > 0:
                            with self._state_lock:
                                self._hops_done += drop
                                hops_done = self._hops_done
                            i += drop
                            continue

                    if target > now:
                        """
                        The Reason: Pacing the Motors.
                            The CPU can process 20ms of audio in less than 1ms. Without this sleep, the robot would try to execute the entire sentence's worth of movement instantly.
                            This "throttles" the loop so that the motor commands are released at the natural human rhythm of the speech.
                        """
                        time.sleep(target - now)
                        with self._state_lock:
                            if self._generation != current_generation:
                                break

                    r = results[i]
                    offsets = (
                        r["x_mm"] / 1000.0,
                        r["y_mm"] / 1000.0,
                        r["z_mm"] / 1000.0,
                        r["roll_rad"],
                        r["pitch_rad"],
                        r["yaw_rad"],
                    )

                    with self._state_lock:
                        if self._generation != current_generation:
                            break

                    self._apply_offsets(offsets)

                    with self._state_lock:
                        self._hops_done += 1
                    i += 1
            finally:
                queue_ref.task_done()
        logger.debug("Head wobbler thread exited")


    def reset(self) -> None:
        """Reset the internal state."""
        with self._state_lock:
            self._generation += 1
            self._base_ts = None
            self._hops_done = 0

        # Drain any queued audio chunks from previous generations
        drained_any = False
        while True:
            try:
                _, _, _ = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            else:
                drained_any = True
                self.audio_queue.task_done()

        with self._sway_lock:
            self.sway.reset()

        if drained_any:
            logger.debug("Head wobbler queue drained during reset")
