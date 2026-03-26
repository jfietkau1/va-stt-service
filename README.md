# VA STT Service

A Python service that captures microphone audio, detects a wake word (OpenWakeWord), transcribes speech (faster-whisper), and sends text events to the VA Orchestrator over WebSocket.

## Architecture

```
Microphone → PyAudio → OpenWakeWord (wake word) → Audio Buffer → faster-whisper → WebSocket → Orchestrator
                                                 → Silero VAD (end-of-speech)
```

### State Machine

1. **IDLE** -- OpenWakeWord processes every audio frame. Low CPU.
2. **LISTENING** -- Wake word detected. Accumulates audio. Sends partial transcripts periodically.
3. **TRANSCRIBING** -- Silence detected (~1.5s). Final transcription pass. Returns to IDLE.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for faster-whisper)
- A working microphone

## Setup

```bash
git clone https://github.com/jfietkau1/va-stt-service.git
cd va-stt-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On first run, OpenWakeWord will automatically download the wake word model.

## Running

```bash
python -m stt_service.main
```

Say "Hey Jarvis" to trigger wake word detection.

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
| `WAKE_WORD_MODEL` | OpenWakeWord model name | `hey_jarvis` |
| `WAKE_THRESHOLD` | Detection threshold 0-1 | `0.5` |
| `WHISPER_MODEL_SIZE` | Model: tiny/base/small/medium/large | `base` |
| `WHISPER_DEVICE` | Device: cpu/cuda/auto | `cuda` |
| `WHISPER_COMPUTE_TYPE` | Compute type: float16/int8/int8_float16 | `float16` |
| `VAD_SILENCE_DURATION` | Seconds of silence to end listening | `1.5` |
| `PARTIAL_TRANSCRIPT_INTERVAL` | Seconds between partial transcripts | `1.0` |
| `WS_HOST` | WebSocket bind address | `0.0.0.0` |
| `WS_PORT` | WebSocket port | `8765` |
| `AUDIO_DEVICE_INDEX` | Mic device index (-1 = default) | `-1` |
