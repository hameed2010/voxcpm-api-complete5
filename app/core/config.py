from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "VoxCPM2 Professional Voice API"
    APP_VERSION: str = "1.0.0"
    API_KEY: str = "change-me"

    VOXCPM_MODEL_ID: str = "openbmb/VoxCPM2"
    WHISPER_MODEL_NAME: str = "large-v3"
    ECAPA_SOURCE: str = "speechbrain/spkrec-ecapa-voxceleb"
    ECAPA_SAVEDIR: str = "storage/models/ecapa_voxceleb"

    DEVICE: str = "cuda"
    REFERENCE_SAMPLE_RATE: int = 16000
    OUTPUT_SAMPLE_RATE: int = 48000
    AUDIO_SAMPLE_RATE: int = 24000
    AUDIO_CHANNELS: int = 1

    MAX_FILE_SIZE_MB: int = 100
    MAX_MERGE_FILES: int = 20

    DEFAULT_CFG: float = 2.0
    DEFAULT_TIMESTEPS: int = 30
    DEFAULT_CANDIDATES: int = 1
    DEFAULT_RETRY_BADCASE: bool = True
    DEFAULT_RETRY_MAX_TIMES: int = 3
    DEFAULT_RETRY_RATIO_THRESHOLD: float = 6.0

    OUTPUT_BITRATE: str = "96k"
    OUTPUT_CODEC: str = "libopus"

    STORAGE_ROOT: str = "storage"
    UPLOAD_DIR: str = "storage/uploads"
    OUTPUT_DIR: str = "storage/outputs"
    TEMP_DIR: str = "storage/temp"

    # API protection / abuse controls
    RATE_LIMIT_REQUESTS: int = 10
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    FFMPEG_TIMEOUT_SECONDS: int = 120

    # RunPod Serverless
    RUNPOD_MODE: bool = False
    ENABLE_WHISPER: bool = True
    ENABLE_ECAPA: bool = True
    MAX_TEXT_LENGTH: int = 8000
    RUNPOD_OUTPUT_AUDIO: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def max_file_size(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    def ensure_directories(self):
        for item in (
            self.STORAGE_ROOT, self.UPLOAD_DIR, self.OUTPUT_DIR,
            self.TEMP_DIR, self.ECAPA_SAVEDIR,
        ):
            Path(item).mkdir(parents=True, exist_ok=True)

    def validate_security(self):
        # RunPod authenticates jobs at the platform boundary. The legacy
        # FastAPI service still uses API_KEY, but Serverless workers do not
        # require a second application-level secret.
        if self.RUNPOD_MODE:
            return
        if not self.API_KEY or self.API_KEY.strip() in {"change-me", "changeme", "your-api-key"}:
            raise RuntimeError(
                "API_KEY is not configured. Set a strong API_KEY in .env before starting the API."
            )

settings = Settings()
