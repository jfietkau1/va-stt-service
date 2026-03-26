import asyncio
import logging
import signal

from .config import settings
from .audio_capture import AudioCapture
from .wake_word import WakeWordDetector
from .transcriber import Transcriber
from .vad import VoiceActivityDetector
from .state_machine import SttStateMachine
from .ws_server import WsServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    if not settings.porcupine_access_key:
        logger.error("PORCUPINE_ACCESS_KEY is not set. Cannot start.")
        return

    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    logger.info("Initializing STT service components...")

    wake_detector = WakeWordDetector(
        access_key=settings.porcupine_access_key,
        keyword=settings.wake_word,
        sensitivity=settings.wake_sensitivity,
    )

    transcriber = Transcriber(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )

    vad = VoiceActivityDetector(
        silence_duration=settings.vad_silence_duration,
    )

    state_machine = SttStateMachine(
        wake_detector=wake_detector,
        transcriber=transcriber,
        vad=vad,
        event_queue=event_queue,
        loop=loop,
        partial_transcript_interval=settings.partial_transcript_interval,
    )

    ws_server = WsServer(
        host=settings.ws_host,
        port=settings.ws_port,
        event_queue=event_queue,
    )

    audio_capture = AudioCapture(
        frame_length=wake_detector.frame_length,
        device_index=settings.audio_device_index,
        frame_callback=state_machine.process_frame,
    )

    await ws_server.start()
    audio_capture.start()

    logger.info("STT service running. Listening for wake word '%s'...", settings.wake_word)

    shutdown_event = asyncio.Event()

    def handle_signal() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    await shutdown_event.wait()

    logger.info("Shutting down...")
    audio_capture.stop()
    await ws_server.stop()
    wake_detector.delete()
    logger.info("STT service stopped")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
