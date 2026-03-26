import logging

import pvporcupine

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Wraps Porcupine for wake word detection on audio frames."""

    def __init__(self, access_key: str, keyword: str = "jarvis", sensitivity: float = 0.5):
        self._porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=[keyword],
            sensitivities=[sensitivity],
        )
        self._keyword = keyword
        logger.info(
            "Porcupine initialized (keyword=%s, frame_length=%d)",
            keyword,
            self.frame_length,
        )

    @property
    def frame_length(self) -> int:
        return self._porcupine.frame_length

    @property
    def sample_rate(self) -> int:
        return self._porcupine.sample_rate

    def process(self, frame: list[int]) -> bool:
        """Process an audio frame. Returns True if the wake word was detected."""
        result = self._porcupine.process(frame)
        if result >= 0:
            logger.info("Wake word '%s' detected", self._keyword)
            return True
        return False

    def delete(self) -> None:
        self._porcupine.delete()
        logger.info("Porcupine resources released")
