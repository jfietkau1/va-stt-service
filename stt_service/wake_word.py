import logging

import numpy as np
from openwakeword.model import Model as OwwModel

logger = logging.getLogger(__name__)

FRAME_LENGTH = 1280
SAMPLE_RATE = 16000


class WakeWordDetector:
    """Wraps OpenWakeWord for wake word detection on audio frames."""

    def __init__(self, wake_word: str = "hey_jarvis", threshold: float = 0.5):
        self._model = OwwModel(
            wakeword_models=[wake_word],
            inference_framework="onnx",
        )
        self._wake_word = wake_word
        self._threshold = threshold
        logger.info(
            "OpenWakeWord initialized (model=%s, threshold=%.2f, frame_length=%d)",
            wake_word,
            threshold,
            self.frame_length,
        )

    @property
    def frame_length(self) -> int:
        return FRAME_LENGTH

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def process(self, frame: list[int] | np.ndarray) -> bool:
        """Process an audio frame. Returns True if the wake word was detected."""
        if isinstance(frame, list):
            audio = np.array(frame, dtype=np.int16)
        else:
            audio = frame

        predictions = self._model.predict(audio)

        for model_name, score in predictions.items():
            if score > self._threshold:
                logger.info("Wake word detected (model=%s, score=%.3f)", model_name, score)
                self._model.reset()
                return True

        return False

    def delete(self) -> None:
        logger.info("OpenWakeWord resources released")
