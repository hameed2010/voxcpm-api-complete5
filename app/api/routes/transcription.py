import uuid
from fastapi import APIRouter, Depends, File, UploadFile
from app.core.security import rate_limit
from app.services.transcription.transcription_service import TranscriptionService

router = APIRouter(prefix="/audio", tags=["Transcription"])

@router.post("/transcribe", dependencies=[Depends(rate_limit)])
async def transcribe(audio: UploadFile = File(...)):
    text = await TranscriptionService().transcribe(audio)
    return {
        "success": True,
        "request_id": uuid.uuid4().hex,
        "model": "large-v3",
        "language": "ar",
        "text": text,
    }
