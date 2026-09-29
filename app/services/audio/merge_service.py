import asyncio
import uuid
from pathlib import Path

from app.core.config import settings
from app.utils.audio import convert_to_wav, merge_audio_files, convert_to_ogg

class AudioMergeService:
    async def merge(self, files):
        if len(files) < 2:
            raise ValueError("At least two audio files are required.")
        if len(files) > settings.MAX_MERGE_FILES:
            raise ValueError(f"Maximum is {settings.MAX_MERGE_FILES} files.")

        normalized = []
        temp_files = []
        merged_wav = None
        output = None
        try:
            for upload in files:
                data = await upload.read(settings.max_file_size + 1)
                if not data:
                    raise ValueError(f"Empty file: {upload.filename}")
                if len(data) > settings.max_file_size:
                    raise ValueError(f"File too large: {upload.filename}")
                raw = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}_raw"
                wav = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}.wav"
                raw.write_bytes(data)
                temp_files.extend([raw, wav])
                await asyncio.to_thread(
                    convert_to_wav, raw, wav,
                    settings.AUDIO_SAMPLE_RATE, settings.AUDIO_CHANNELS
                )
                normalized.append(wav)

            merged_wav = Path(settings.TEMP_DIR) / f"merged_{uuid.uuid4().hex}.wav"
            output = Path(settings.OUTPUT_DIR) / f"merged_{uuid.uuid4().hex}.ogg"
            temp_files.append(merged_wav)

            await asyncio.to_thread(
                merge_audio_files, normalized, merged_wav,
                settings.AUDIO_SAMPLE_RATE, settings.AUDIO_CHANNELS
            )
            await asyncio.to_thread(
                convert_to_ogg, merged_wav, output,
                settings.OUTPUT_BITRATE, settings.OUTPUT_SAMPLE_RATE
            )
            return output
        finally:
            for f in temp_files:
                Path(f).unlink(missing_ok=True)
