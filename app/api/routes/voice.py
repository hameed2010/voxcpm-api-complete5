from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.security import require_api_key, rate_limit
from app.services.voice.clone_service import VoiceCloneService

router = APIRouter(prefix="/voice", tags=["Voice Cloning"])

@router.post("/clone", dependencies=[Depends(rate_limit)])
async def clone_voice(
    request: Request,
    reference_audio: UploadFile = File(...),
    target_text: str = Form(...),
    prompt_text: str | None = Form(None),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(30),
    candidates: int = Form(1),
    seed: int | None = Form(None),
    retry_badcase: bool = Form(True),
    retry_max_times: int = Form(3),
    retry_ratio_threshold: float = Form(6.0),
    auto_transcribe: bool = Form(False),
    silence_ms: int = Form(350),
    single_pass: bool = Form(True),
    auto_sentence_points: bool = Form(True),
    enable_reference_rhythm: bool = Form(True),
    rhythm_strength: float = Form(1.0),
):
    if candidates < 1:
        raise HTTPException(status_code=400, detail="candidates must be >= 1")
    if candidates != 1:
        # Project policy: single candidate for predictable GPU usage.
        raise HTTPException(status_code=400, detail="candidates must be 1")
    if settings.MAX_FILE_SIZE_MB <= 0:
        raise HTTPException(status_code=500, detail="Invalid MAX_FILE_SIZE_MB configuration")

    result = await VoiceCloneService().clone(
        reference_audio, prompt_text, target_text,
        cfg_value=cfg_value, inference_timesteps=inference_timesteps,
        candidates=1, seed=seed, retry_badcase=retry_badcase,
        retry_max_times=retry_max_times,
        retry_ratio_threshold=retry_ratio_threshold,
        auto_transcribe=auto_transcribe,
        chunk_long_text="؟" in (target_text or ""),
        silence_ms=silence_ms, single_pass=single_pass,
        auto_sentence_points=auto_sentence_points,
        enable_reference_rhythm=enable_reference_rhythm,
        rhythm_strength=rhythm_strength,
    )
    base = str(request.base_url).rstrip("/")
    return {
        "success": True, "request_id": result["request_id"],
        "format": "ogg", "mime_type": "audio/ogg",
        "audio_url": f"{base}/api/v1/voice/audio/{Path(result['output']).name}",
        "prompt_text": result["prompt_text"], "seed": result["seed"],
        "duration": result["duration"], "speaker_score": result["speaker_score"],
        "same_speaker": result["same_speaker"], "reference_qc": result["reference_qc"],
        "chunks_count": result["chunks_count"], "single_pass": result["single_pass"],
        "auto_sentence_points": result["auto_sentence_points"],
        "enable_reference_rhythm": result["enable_reference_rhythm"],
        "rhythm_strength": result["rhythm_strength"],
        "final_target_text": result["final_target_text"],
        "ranking": result["ranking"],
        "candidates": 1,
    }

@router.get("/audio/{filename}", dependencies=[Depends(require_api_key)])
async def get_voice_audio(filename: str):
    root = Path(settings.OUTPUT_DIR).resolve()
    safe_name = Path(filename).name
    path = (root / safe_name).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/ogg", filename=path.name)
