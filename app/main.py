from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.lifecycle import lifespan
from app.core.logging import configure_logging
from app.api.routes import health, voice, transcription, audio

configure_logging()
settings.ensure_directories()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Professional API for VoxCPM2 voice cloning, Arabic transcription, "
        "speaker similarity scoring, audio merging and OGG/Opus output."
    ),
    lifespan=lifespan,
)

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"success": False, "detail": str(exc)},
    )

app.include_router(health.router, prefix="/api/v1")
app.include_router(voice.router, prefix="/api/v1")
app.include_router(transcription.router, prefix="/api/v1")
app.include_router(audio.router, prefix="/api/v1")
