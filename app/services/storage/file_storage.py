from pathlib import Path
import shutil
import uuid

class FileStorage:
    def __init__(self, upload_dir, output_dir, temp_dir):
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        for directory in (self.upload_dir, self.output_dir, self.temp_dir):
            directory.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, upload_file, max_size):
        suffix = Path(upload_file.filename or "").suffix.lower()
        data = await upload_file.read()
        if not data:
            raise ValueError(f"Empty file: {upload_file.filename}")
        if len(data) > max_size:
            raise ValueError(f"File exceeds maximum size: {max_size // 1024 // 1024} MB")
        target = self.upload_dir / f"{uuid.uuid4().hex}{suffix}"
        target.write_bytes(data)
        return target

    def output_path(self, extension="ogg"):
        return self.output_dir / f"{uuid.uuid4().hex}.{extension}"

    def temp_path(self, extension="wav"):
        return self.temp_dir / f"{uuid.uuid4().hex}.{extension}"
