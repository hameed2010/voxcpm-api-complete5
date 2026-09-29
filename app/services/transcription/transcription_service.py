import asyncio
import torch
from pathlib import Path
import uuid

from app.core.model_manager import model_manager
from app.core.config import settings
from app.utils.audio import convert_to_wav

class TranscriptionService:
    async def transcribe(self, upload_file):
        raw = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}_input"
        wav = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}.wav"
        try:
            data = await upload_file.read(settings.max_file_size + 1)
            if not data:
                raise ValueError("Audio file is empty.")
            if len(data) > settings.max_file_size:
                raise ValueError("Audio file exceeds the maximum allowed size.")
            raw.write_bytes(data)
            await asyncio.to_thread(
                convert_to_wav, raw, wav,
                settings.AUDIO_SAMPLE_RATE, settings.AUDIO_CHANNELS
            )
            return await self.transcribe_file(wav)
        finally:
            raw.unlink(missing_ok=True)
            wav.unlink(missing_ok=True)

    async def transcribe_file(self, wav_path):
        await model_manager.ensure_whisper()
        async with model_manager.model_semaphore:
            return await asyncio.to_thread(self._transcribe_sync, wav_path)

    def _transcribe_sync(self, wav_path):
        result = model_manager.whisper.transcribe(
            str(wav_path), language="ar", task="transcribe",
            fp16=torch.cuda.is_available(), temperature=0, verbose=False,
        )
        return result.get("text", "").strip()
