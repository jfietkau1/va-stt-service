# VA STT Service

A Python service that captures microphone audio, detects a wake word (Porcupine), transcribes speech (faster-whisper), and sends text events to the VA Orchestrator over WebSocket.

## Architecture

```
Microphone → PvRecorder → Porcupine (wake word) → Audio Buffer → faster-whisper → WebSocket → Orchestrator
                                                 → Silero VAD (end-of-speech)
```

### State Machine

1. **IDLE** -- Porcupine processes every audio frame. Low CPU.
2. **LISTENING** -- Wake word detected. Accumulates audio. Sends partial transcripts periodically.
3. **TRANSCRIBING** -- Silence detected (~1.5s). Final transcription pass. Returns to IDLE.

## Prerequisites

- Python 3.10+
- A Picovoice AccessKey (free at https://console.picovoice.ai/)
- CUDA-capable GPU (recommended for faster-whisper)

## Setup

```bash
git clone https://github.com/jfietkau1/va-stt-service.git
cd va-stt-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Porcupine access key
```

## Running

```bash
python -m stt_service.main
```

## WebSocket Protocol

Messages sent to the orchestrator (port 8765):

| Message | Description |
|---|---|
| `{ "type": "wake" }` | Wake word detected, listening started |
| `{ "type": "transcript", "text": "...", "final": false }` | Partial transcription |
| `{ "type": "transcript", "text": "...", "final": true }` | Final transcription |
| `{ "type": "silence" }` | User stopped speaking |

## Configuration

| Variable | Description | Default |
|---|---|---|
| `PORCUPINE_ACCESS_KEY` | Picovoice access key | (required) |
| `WAKE_WORD` | Wake word to listen for | `jarvis` |
| `WAKE_SENSITIVITY` | Detection sensitivity 0-1 | `0.5` |
| `WHISPER_MODEL_SIZE` | Model: tiny/base/small/medium/large | `base` |
| `WHISPER_DEVICE` | Device: cpu/cuda/auto | `cuda` |
| `WHISPER_COMPUTE_TYPE` | Compute type: float16/int8/int8_float16 | `float16` |
| `VAD_SILENCE_DURATION` | Seconds of silence to end listening | `1.5` |
| `PARTIAL_TRANSCRIPT_INTERVAL` | Seconds between partial transcripts | `1.0` |
| `WS_HOST` | WebSocket bind address | `0.0.0.0` |
| `WS_PORT` | WebSocket port | `8765` |
| `AUDIO_DEVICE_INDEX` | Mic device index (-1 = default) | `-1` |
