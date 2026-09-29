from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.security import require_api_key, rate_limit
from app.services.audio.merge_service import AudioMergeService

router = APIRouter(prefix="/audio", tags=["Audio"])

@router.post("/merge", dependencies=[Depends(rate_limit)])
async def merge(files: List[UploadFile] = File(...), request: Request = None):
    output = await AudioMergeService().merge(files)
    base = str(request.base_url).rstrip("/")
    return {
        "success": True,
        "request_id": Path(output).stem,
        "format": "ogg",
        "mime_type": "audio/ogg",
        "audio_url": f"{base}/api/v1/audio/output/{Path(output).name}",
        "input_count": len(files),
    }

@router.get("/output/{filename}", dependencies=[Depends(require_api_key)])
async def get_output(filename: str):
    root = Path(settings.OUTPUT_DIR).resolve()
    safe_name = Path(filename).name
    path = (root / safe_name).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/ogg", filename=path.name)
