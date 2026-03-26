import asyncio
import enum
import logging
import time

import numpy as np

from .transcriber import Transcriber
from .vad import VoiceActivityDetector
from .wake_word import WakeWordDetector

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"


class SttStateMachine:
    """Manages the IDLE -> LISTENING -> TRANSCRIBING pipeline.

    Audio frames are fed in from the capture thread via `process_frame()`.
    Events are dispatched to connected WebSocket clients via an asyncio queue.
    """

    def __init__(
        self,
        wake_detector: WakeWordDetector,
        transcriber: Transcriber,
        vad: VoiceActivityDetector,
        event_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        partial_transcript_interval: float = 1.0,
    ):
        self._wake = wake_detector
        self._transcriber = transcriber
        self._vad = vad
        self._event_queue = event_queue
        self._loop = loop
        self._partial_interval = partial_transcript_interval

        self._state = State.IDLE
        self._audio_buffer: list[int] = []
        self._last_partial_time: float = 0.0
        self._frame_duration: float = wake_detector.frame_length / SAMPLE_RATE

    @property
    def state(self) -> State:
        return self._state

    def process_frame(self, frame: list[int]) -> None:
        """Called from the audio capture thread for every mic frame."""
        if self._state == State.IDLE:
            self._handle_idle(frame)
        elif self._state == State.LISTENING:
            self._handle_listening(frame)

    def _handle_idle(self, frame: list[int]) -> None:
        if self._wake.process(frame):
            self._state = State.LISTENING
            self._audio_buffer = list(frame)
            self._last_partial_time = time.monotonic()
            self._vad.reset()
            self._emit({"type": "wake"})
            logger.info("State: IDLE -> LISTENING")

    def _handle_listening(self, frame: list[int]) -> None:
        self._audio_buffer.extend(frame)

        is_silence = self._vad.process(frame, self._frame_duration)

        now = time.monotonic()
        if now - self._last_partial_time >= self._partial_interval:
            self._last_partial_time = now
            self._run_partial_transcription()

        if is_silence:
            self._emit({"type": "silence"})
            self._state = State.TRANSCRIBING
            logger.info("State: LISTENING -> TRANSCRIBING")
            self._run_final_transcription()

    def _run_partial_transcription(self) -> None:
        audio = np.array(self._audio_buffer, dtype=np.int16)
        if len(audio) < SAMPLE_RATE * 0.5:
            return

        text = self._transcriber.transcribe(audio, SAMPLE_RATE)
        if text:
            self._emit({"type": "transcript", "text": text, "final": False})

    def _run_final_transcription(self) -> None:
        audio = np.array(self._audio_buffer, dtype=np.int16)
        self._audio_buffer = []

        if len(audio) < SAMPLE_RATE * 0.3:
            logger.debug("Audio too short for transcription, skipping")
            self._state = State.IDLE
            return

        text = self._transcriber.transcribe(audio, SAMPLE_RATE)
        if text:
            self._emit({"type": "transcript", "text": text, "final": True})
        else:
            logger.debug("Empty transcription result")

        self._state = State.IDLE
        logger.info("State: TRANSCRIBING -> IDLE")

    def _emit(self, event: dict) -> None:
        self._loop.call_soon_threadsafe(self._event_queue.put_nowait, event)
