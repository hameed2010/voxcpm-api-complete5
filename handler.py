import asyncio
import base64
import binascii
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

import runpod

from app.core.config import settings
from app.core.model_manager import model_manager
from app.services.voice.clone_service import VoiceCloneService
from app.services.transcription.transcription_service import TranscriptionService
from app.utils.audio import convert_to_wav

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("voxcpm.runpod")

# Serverless worker is intentionally single-GPU / single-heavy-job.
_JOB_LOCK = asyncio.Lock()


class BytesUpload:
    """Small UploadFile-compatible object used by existing service code."""

    def __init__(self, data: bytes, filename: str = "reference.wav"):
        self.data = data
        self.filename = filename

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self.data
        return self.data[:size]


def _clean_b64(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reference_audio_base64 is required.")
    value = value.strip()
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    value = "".join(value.split())
    return value


def decode_audio(input_data: dict) -> bytes:
    encoded = input_data.get("reference_audio_base64")
    if not encoded:
        # Allow a few common aliases for easier Laravel/n8n integration.
        encoded = input_data.get("audio_base64") or input_data.get("reference_audio")
    if not encoded:
        raise ValueError(
            "Missing reference audio. Send reference_audio_base64 "
            "(base64 or data URI)."
        )
    try:
        raw = base64.b64decode(_clean_b64(encoded), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 audio.") from exc

    if not raw:
        raise ValueError("Reference audio is empty.")
    if len(raw) > settings.max_file_size:
        raise ValueError(
            f"Reference audio exceeds {settings.MAX_FILE_SIZE_MB} MB."
        )
    return raw


def encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _int(value, default):
    return default if value is None else int(value)


def _float(value, default):
    return default if value is None else float(value)


def _bool(value, default):
    return default if value is None else bool(value)


async def clone_job(inp: dict) -> dict:
    target_text = str(inp.get("target_text") or "").strip()
    prompt_text = inp.get("prompt_text")
    prompt_text = str(prompt_text).strip() if prompt_text is not None else None

    if not target_text:
        raise ValueError("target_text is required.")
    if len(target_text) > settings.MAX_TEXT_LENGTH:
        raise ValueError(f"target_text exceeds {settings.MAX_TEXT_LENGTH} characters.")

    if prompt_text and len(prompt_text) > settings.MAX_TEXT_LENGTH:
        raise ValueError(f"prompt_text exceeds {settings.MAX_TEXT_LENGTH} characters.")

    raw = decode_audio(inp)

    # Keep one request at a time for deterministic GPU memory usage.
    async with _JOB_LOCK:
        upload = BytesUpload(raw)
        result = await VoiceCloneService().clone(
            upload,
            prompt_text,
            target_text,
            cfg_value=_float(inp.get("cfg_value"), settings.DEFAULT_CFG),
            inference_timesteps=_int(
                inp.get("inference_timesteps"), settings.DEFAULT_TIMESTEPS
            ),
            candidates=1,
            seed=inp.get("seed"),
            retry_badcase=_bool(
                inp.get("retry_badcase"), settings.DEFAULT_RETRY_BADCASE
            ),
            retry_max_times=_int(
                inp.get("retry_max_times"), settings.DEFAULT_RETRY_MAX_TIMES
            ),
            retry_ratio_threshold=_float(
                inp.get("retry_ratio_threshold"),
                settings.DEFAULT_RETRY_RATIO_THRESHOLD,
            ),
            auto_transcribe=_bool(inp.get("auto_transcribe"), False),
            chunk_long_text=_bool(inp.get("chunk_long_text"), False),
            silence_ms=_int(inp.get("silence_ms"), 350),
            single_pass=_bool(inp.get("single_pass"), True),
            auto_sentence_points=_bool(inp.get("auto_sentence_points"), True),
            enable_reference_rhythm=_bool(
                inp.get("enable_reference_rhythm"), True
            ),
            rhythm_strength=max(
                0.0, min(1.0, _float(inp.get("rhythm_strength"), 1.0))
            ),
        )

        output = Path(result["output"])
        response = {
            "success": True,
            "request_id": result["request_id"],
            "format": "ogg",
            "mime_type": "audio/ogg",
            "prompt_text": result["prompt_text"],
            "seed": result["seed"],
            "duration": result["duration"],
            "speaker_score": result["speaker_score"],
            "same_speaker": result["same_speaker"],
            "reference_qc": result["reference_qc"],
            "chunks_count": result["chunks_count"],
            "single_pass": result["single_pass"],
            "auto_sentence_points": result["auto_sentence_points"],
            "enable_reference_rhythm": result["enable_reference_rhythm"],
            "rhythm_strength": result["rhythm_strength"],
            "final_target_text": result["final_target_text"],
            "ranking": result["ranking"],
            "candidates": 1,
        }

        if _bool(inp.get("return_audio_base64"), settings.RUNPOD_OUTPUT_AUDIO):
            response["audio_base64"] = encode_file(output)

        return response


async def transcribe_job(inp: dict) -> dict:
    raw = decode_audio(inp)
    suffix = str(inp.get("extension") or ".audio")
    if not suffix.startswith("."):
        suffix = "." + suffix

    async with _JOB_LOCK:
        temp_raw = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}{suffix}"
        temp_wav = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}.wav"
        try:
            temp_raw.write_bytes(raw)
            await asyncio.to_thread(
                convert_to_wav,
                temp_raw,
                temp_wav,
                settings.AUDIO_SAMPLE_RATE,
                settings.AUDIO_CHANNELS,
            )
            text = await TranscriptionService().transcribe_file(temp_wav)
            return {"success": True, "text": text}
        finally:
            temp_raw.unlink(missing_ok=True)
            temp_wav.unlink(missing_ok=True)


async def async_handler(job: dict) -> dict:
    job_input = job.get("input") or {}
    if not isinstance(job_input, dict):
        raise ValueError("input must be an object.")

    action = str(job_input.get("action") or "clone").lower().strip()

    if action == "health":
        return {
            "success": model_manager.ready,
            "status": "healthy" if model_manager.ready else "starting",
            "models": {
                "voxcpm2": model_manager.voxcpm is not None,
                "whisper": model_manager.whisper is not None,
                "ecapa": model_manager.speaker_verification is not None,
            },
        }

    if action == "clone":
        return await clone_job(job_input)

    if action == "transcribe":
        return await transcribe_job(job_input)

    raise ValueError(f"Unsupported action: {action}")


def handler(job: dict) -> dict:
    """RunPod synchronous handler entry point."""
    try:
        return asyncio.run(async_handler(job))
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        return {"success": False, "error": str(exc), "error_type": "validation"}
    except Exception as exc:
        logger.exception("RunPod job failed")
        return {
            "success": False,
            "error": str(exc),
            "error_type": "runtime",
        }


@runpod.serverless.register_fitness_check
def check_gpu():
    """Fail fast if the Serverless worker was assigned an unusable GPU."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for VoxCPM2 Serverless.")
    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / (1024 ** 3)
    if memory_gb < 12:
        raise RuntimeError(
            f"GPU VRAM is too small: {memory_gb:.1f} GB. "
            "Use a GPU with at least 12 GB VRAM."
        )


if __name__ == "__main__":
    # Core models are loaded once per warm worker, not once per request.
    asyncio.run(model_manager.load_all())
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": lambda current: 1,
    })
