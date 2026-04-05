"""Bidirectional local audio stream for the Reachy Mini Teacher App.

Supports both Gemini Live and Local Pipeline modes by accepting any
AsyncStreamHandler-compatible handler instance.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend


logger = logging.getLogger(__name__)


class LocalStream:
    """Drive the robot's mic/speaker through any AsyncStreamHandler handler."""

    def __init__(
        self,
        handler: Any,  # GeminiLiveHandler | LocalPipelineHandler
        robot: ReachyMini,
    ) -> None:
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        # Allow the handler to flush the player queue
        self.handler._clear_queue = self.clear_audio_queue

    # ------------------------------------------------------------------

    def launch(self) -> None:
        """Start mic/speaker pipelines and run the async event loops."""
        self._stop_event.clear()
        self._robot.media.start_recording()
        self._robot.media.start_playing()

        import time
        time.sleep(1)  # let pipelines stabilise

        asyncio.run(self._runner())

    def close(self) -> None:
        """Gracefully stop recording, playback, and all async tasks."""
        logger.info("Stopping LocalStream …")
        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug("stop_recording: %s", e)
        try:
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug("stop_playing: %s", e)

        self._stop_event.set()
        for task in self._tasks:
            if not task.done():
                task.cancel()

    def clear_audio_queue(self) -> None:
        """Flush the player's audio buffer immediately."""
        logger.info("Flushing audio queue")
        if self._robot.media.backend == MediaBackend.GSTREAMER:
            self._robot.media.audio.clear_player()
        else:
            self._robot.media.audio.clear_output_buffer()
        self.handler.output_queue = asyncio.Queue()

    # ------------------------------------------------------------------

    async def _runner(self) -> None:
        self._tasks = [
            asyncio.create_task(self.handler.start_up(), name="handler-startup"),
            asyncio.create_task(self._record_loop(), name="record-loop"),
            asyncio.create_task(self._play_loop(), name="play-loop"),
        ]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled during shutdown")
        finally:
            await self.handler.shutdown()

    async def _record_loop(self) -> None:
        """Forward mic frames from the robot to the handler."""
        input_sr = self._robot.media.get_input_audio_samplerate()
        logger.info("Record loop started — input_sr=%d Hz", input_sr)

        frames_received = 0
        zero_sr_logged = False

        while not self._stop_event.is_set():
            # Re-fetch sample rate in case WebRTC audio initialises after startup
            current_sr = self._robot.media.get_input_audio_samplerate()
            if current_sr != input_sr:
                input_sr = current_sr
                logger.info("Input sample rate updated to %d Hz", input_sr)

            if input_sr == 0:
                if not zero_sr_logged:
                    logger.warning(
                        "Input sample rate is 0 — robot audio not ready yet. "
                        "Check that the reachy-mini daemon is running on the robot "
                        "and the WebRTC media pipeline has connected."
                    )
                    zero_sr_logged = True
                await asyncio.sleep(0.1)
                continue

            zero_sr_logged = False
            frame = self._robot.media.get_audio_sample()
            if frame is not None:
                frames_received += 1
                if frames_received == 1:
                    logger.info("First audio frame received from robot mic (input_sr=%d Hz)", input_sr)
                await self.handler.receive((input_sr, frame))
            await asyncio.sleep(0)

    async def _play_loop(self) -> None:
        """Fetch output from the handler and play it or log it."""
        while not self._stop_event.is_set():
            output = await self.handler.emit()

            if isinstance(output, AdditionalOutputs):
                for msg in output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            msg.get("role"),
                            content[:500] + ("…" if len(content) > 500 else ""),
                        )

            elif isinstance(output, tuple):
                input_sr, audio_data = output
                output_sr = self._robot.media.get_output_audio_samplerate()

                if audio_data.ndim == 2:
                    if audio_data.shape[1] > audio_data.shape[0]:
                        audio_data = audio_data.T
                    if audio_data.shape[1] > 1:
                        audio_data = audio_data[:, 0]

                audio_frame = audio_to_float32(audio_data)

                if len(audio_frame) == 0:
                    await asyncio.sleep(0)
                    continue

                if input_sr != output_sr and output_sr > 0:
                    num_samples = int(len(audio_frame) * output_sr / input_sr)
                    if num_samples > 0:
                        audio_frame = resample(audio_frame, num_samples)

                self._robot.media.push_audio_sample(audio_frame)

            await asyncio.sleep(0)

