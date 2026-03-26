import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class VoiceActivityDetector:
    """Silero VAD wrapper for detecting end-of-speech.

    Processes audio frames and tracks consecutive silence duration.
    Reports whether the user has stopped speaking based on a configurable
    silence threshold.
    """

    def __init__(self, silence_duration: float = 1.5):
        self._silence_duration = silence_duration
        self._model, self._utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
        self._consecutive_silence: float = 0.0
        self._has_speech = False
        logger.info(
            "Silero VAD loaded (silence_threshold=%.1fs)",
            silence_duration,
        )

    def reset(self) -> None:
        self._consecutive_silence = 0.0
        self._has_speech = False
        self._model.reset_states()

    def process(self, frame: list[int] | np.ndarray, frame_duration: float) -> bool:
        """Process an audio frame.

        Returns True if silence has been detected for longer than the threshold
        (meaning the user stopped speaking).
        """
        if isinstance(frame, list):
            audio = np.array(frame, dtype=np.float32) / 32768.0
        elif frame.dtype == np.int16:
            audio = frame.astype(np.float32) / 32768.0
        else:
            audio = frame

        tensor = torch.from_numpy(audio)
        speech_prob = self._model(tensor, SAMPLE_RATE).item()

        if speech_prob > 0.5:
            self._has_speech = True
            self._consecutive_silence = 0.0
        else:
            self._consecutive_silence += frame_duration

        if self._has_speech and self._consecutive_silence >= self._silence_duration:
            logger.debug(
                "End of speech detected (%.1fs silence)",
                self._consecutive_silence,
            )
            return True

        return False
