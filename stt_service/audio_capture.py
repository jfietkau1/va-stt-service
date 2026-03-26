import logging
import threading
from collections.abc import Callable

import pyaudio
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1


class AudioCapture:
    """Captures audio from the microphone using PyAudio in a background thread.

    Calls `frame_callback` with each audio frame (list of int16 samples).
    """

    def __init__(
        self,
        frame_length: int,
        device_index: int = -1,
        frame_callback: Callable[[list[int]], None] | None = None,
    ):
        self._frame_length = frame_length
        self._device_index = device_index if device_index >= 0 else None
        self._frame_callback = frame_callback
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=self._frame_length,
            input_device_index=self._device_index,
        )
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        device_info = "default"
        if self._device_index is not None:
            device_info = str(self._device_index)
        logger.info(
            "Audio capture started (device=%s, frame_length=%d)",
            device_info,
            self._frame_length,
        )

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Audio capture stopped")

    def _capture_loop(self) -> None:
        while self._running and self._stream:
            try:
                raw = self._stream.read(self._frame_length, exception_on_overflow=False)
                frame = np.frombuffer(raw, dtype=np.int16).tolist()
                if self._frame_callback:
                    self._frame_callback(frame)
            except Exception:
                if self._running:
                    logger.exception("Error reading audio frame")
                break
