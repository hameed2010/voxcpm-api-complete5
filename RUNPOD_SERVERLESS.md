# VoxCPM2 — RunPod Serverless

## Input

Use a RunPod job with:

```json
{
  "input": {
    "action": "clone",
    "reference_audio_base64": "<BASE64_AUDIO>",
    "prompt_text": "النص المطابق للعينة الصوتية",
    "target_text": "النص الذي تريد توليده",
    "cfg_value": 2.0,
    "inference_timesteps": 30,
    "seed": 12345,
    "retry_badcase": true,
    "retry_max_times": 3,
    "retry_ratio_threshold": 6.0,
    "single_pass": true,
    "auto_sentence_points": true,
    "enable_reference_rhythm": true,
    "rhythm_strength": 1.0,
    "return_audio_base64": true
  }
}
```

`reference_audio_base64` accepts plain base64 or a `data:audio/...;base64,...` data URI.

## Actions

- `clone`: VoxCPM2 voice cloning.
- `transcribe`: Arabic Whisper transcription.
- `health`: worker/model state.

## Important

The worker loads VoxCPM2 and ECAPA once and reuses them for warm jobs. Whisper is loaded lazily only when transcription/`auto_transcribe` is requested.

Heavy GPU inference is serialized per worker. This is intentional: one GPU worker should not run two VoxCPM generations concurrently and risk VRAM exhaustion.

## Docker

Build:

```bash
docker build -f Dockerfile.serverless -t YOUR_DOCKERHUB/voxcpm2-serverless:1.0.0 .
```

Push:

```bash
docker push YOUR_DOCKERHUB/voxcpm2-serverless:1.0.0
```

Then create a RunPod Serverless endpoint using that image.

Recommended starting point for a 48 GB GPU:
- GPU: A40 / A6000-class 48 GB
- Workers: 0 minimum for scale-to-zero, 1 max while testing
- Max concurrency per worker: 1
- Idle timeout: 5–15 minutes
- Execution timeout: set above your longest expected generation

A network volume is recommended for Hugging Face/Torch caches so repeated worker starts do not redownload model assets.

## Laravel / n8n

Send JSON to RunPod's `/runsync` endpoint for short jobs or `/run` for asynchronous jobs. Put the RunPod API key in the Authorization header and send the `input` object shown above.

For production, keep the RunPod API key server-side. Do not expose it in Flutter/browser code.


## Why this is a real Serverless worker

The container starts `handler.py`, and `handler.py` registers the RunPod
`handler(job)` with `runpod.serverless.start(...)`. It does not start
Uvicorn/HTTP inside the worker. This is the standard RunPod Serverless worker
pattern. The worker is explicitly limited to one concurrent heavy job per GPU.
