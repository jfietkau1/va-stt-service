import logging
import threading
from collections.abc import Callable

from pvrecorder import PvRecorder

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from the microphone using PvRecorder in a background thread.

    Calls `frame_callback` with each audio frame (list of int16 samples).
    Frame size is determined by `frame_length` (must match Porcupine's requirement).
    """

    def __init__(
        self,
        frame_length: int,
        device_index: int = -1,
        frame_callback: Callable[[list[int]], None] | None = None,
    ):
        self._frame_length = frame_length
        self._device_index = device_index
        self._frame_callback = frame_callback
        self._recorder: PvRecorder | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return

        self._recorder = PvRecorder(
            frame_length=self._frame_length,
            device_index=self._device_index,
        )
        self._running = True
        self._recorder.start()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Audio capture started (device=%s, frame_length=%d)",
            self._recorder.selected_device,
            self._frame_length,
        )

    def stop(self) -> None:
        self._running = False
        if self._recorder:
            self._recorder.stop()
            self._recorder.delete()
            self._recorder = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Audio capture stopped")

    def _capture_loop(self) -> None:
        while self._running and self._recorder:
            try:
                frame = self._recorder.read()
                if self._frame_callback:
                    self._frame_callback(frame)
            except Exception:
                if self._running:
                    logger.exception("Error reading audio frame")
                break
