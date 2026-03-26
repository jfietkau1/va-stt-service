import logging

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class Transcriber:
    """Wraps faster-whisper for batch transcription of accumulated audio buffers."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        logger.info(
            "Loading Whisper model (size=%s, device=%s, compute_type=%s)...",
            model_size, device, compute_type,
        )
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper model loaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe an audio buffer (int16 or float32 numpy array).

        Returns the concatenated text of all segments.
        """
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        segments, info = self._model.transcribe(
            audio,
            beam_size=5,
            language=None,
            vad_filter=False,
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        result = " ".join(text_parts).strip()
        logger.debug(
            "Transcribed %.1fs of audio (lang=%s, prob=%.2f): %s",
            info.duration,
            info.language,
            info.language_probability,
            result[:100],
        )
        return result
