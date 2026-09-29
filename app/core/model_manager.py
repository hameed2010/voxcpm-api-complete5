import asyncio
import inspect
import logging
import threading

import torch
import whisper
from voxcpm import VoxCPM
from speechbrain.inference.speaker import SpeakerRecognition

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Process-local singleton for GPU models.

    RunPod Serverless keeps a worker process alive across jobs, so models are
    loaded once and reused. Heavy inference is serialized by model_semaphore
    to prevent concurrent GPU OOM on a single worker.
    """

    def __init__(self):
        self.voxcpm = None
        self.whisper = None
        self.speaker_verification = None
        self.ready = False
        self.whisper_ready = False
        self.ecapa_ready = False
        self.model_semaphore = asyncio.Semaphore(1)
        self._load_lock = threading.Lock()

    @property
    def device(self) -> str:
        configured = settings.DEVICE
        if configured.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to CPU.")
            return "cpu"
        return configured

    def _load_voxcpm(self):
        if self.voxcpm is not None:
            return
        logger.info("Loading VoxCPM2: %s on %s", settings.VOXCPM_MODEL_ID, self.device)
        sig = inspect.signature(VoxCPM.from_pretrained)
        kwargs = {"load_denoiser": False}
        if "device" in sig.parameters:
            kwargs["device"] = self.device
        self.voxcpm = VoxCPM.from_pretrained(settings.VOXCPM_MODEL_ID, **kwargs)
        logger.info("VoxCPM2 ready")

    def _load_ecapa(self):
        if self.ecapa_ready or not settings.ENABLE_ECAPA:
            return
        logger.info("Loading ECAPA: %s", settings.ECAPA_SOURCE)
        self.speaker_verification = SpeakerRecognition.from_hparams(
            source=settings.ECAPA_SOURCE,
            savedir=settings.ECAPA_SAVEDIR,
            run_opts={"device": self.device},
        )
        self.ecapa_ready = True
        logger.info("ECAPA ready")

    def _load_whisper(self):
        if self.whisper_ready or not settings.ENABLE_WHISPER:
            return
        logger.info("Loading Whisper: %s on %s", settings.WHISPER_MODEL_NAME, self.device)
        self.whisper = whisper.load_model(settings.WHISPER_MODEL_NAME, device=self.device)
        self.whisper_ready = True
        logger.info("Whisper ready")

    async def load_all(self):
        if self.ready:
            return
        settings.ensure_directories()
        settings.validate_security()
        with self._load_lock:
            if self.voxcpm is None:
                self._load_voxcpm()
            if settings.ENABLE_ECAPA and not self.ecapa_ready:
                self._load_ecapa()
        self.ready = self.voxcpm is not None
        logger.info("Core AI models ready")

    async def ensure_whisper(self):
        if not self.ready:
            await self.load_all()
        if not self.whisper_ready:
            async with self.model_semaphore:
                await asyncio.to_thread(self._load_whisper)

    async def unload_all(self):
        self.voxcpm = None
        self.whisper = None
        self.speaker_verification = None
        self.ready = False
        self.whisper_ready = False
        self.ecapa_ready = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


model_manager = ModelManager()
