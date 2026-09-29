import inspect
import asyncio
import uuid
import re
from pathlib import Path
import numpy as np
import soundfile as sf
from app.core.config import settings
from app.core.model_manager import model_manager
from app.utils.audio import (
    normalize_peak, prepare_reference, reference_quality_report,
    normalize_iraqi_target, convert_to_ogg,
    split_text_at_question_marks, concatenate_with_silence,
)

class VoiceCloneService:
    def _speaker_score(self, reference_path, generated_path):
        if model_manager.speaker_verification is None:
            return {"score": None, "same_speaker": None}
        score, prediction = model_manager.speaker_verification.verify_files(
            str(reference_path), str(generated_path)
        )
        return {
            "score": float(score.squeeze().cpu().item()),
            "same_speaker": int(prediction.squeeze().cpu().item()) == 1,
        }

    async def _speaker_score_async(self, reference_path, generated_path):
        async with model_manager.model_semaphore:
            return await asyncio.to_thread(
                self._speaker_score, reference_path, generated_path
            )

    @staticmethod
    def _auto_add_sentence_points(text, min_words=5, max_words=8):
        """Add punctuation boundaries without splitting the audio generation.

        The complete target remains one string and is sent to VoxCPM2 in one
        model.generate() call. Existing punctuation is preserved.
        """
        if not text:
            return ""
        text = re.sub(r"\s+", " ", str(text).strip())
        if not text:
            return ""

        words = text.split()
        out, chunk = [], []
        # Six words is deliberately inside the requested 5-8 word range.
        target_words = 6
        for word in words:
            chunk.append(word)
            if re.search(r"[.!؟?]$", word):
                out.append(" ".join(chunk))
                chunk = []
            elif len(chunk) >= target_words:
                out.append(" ".join(chunk) + ".")
                chunk = []

        if chunk:
            last = " ".join(chunk)
            if not re.search(r"[.!؟?]$", last):
                last += "."
            out.append(last)
        return " ".join(out)

    @staticmethod
    def _text_units(text):
        text = re.sub(r"\s+", "", str(text or ""))
        text = re.sub(r"[\u061F\?\!\.،,؛;:\"'“”‘’()\[\]{}]", "", text)
        return max(1, len(text))

    def _fit_audio_to_reference_rate(self, wav, reference_duration, reference_text,
                                     target_text, strength=1.0,
                                     min_rate=0.78, max_rate=1.28):
        """Mild pitch-preserving rate correction based on the full reference."""
        import librosa
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        sr = int(getattr(model_manager.voxcpm.tts_model, "sample_rate", settings.OUTPUT_SAMPLE_RATE))
        if wav.size < int(sr * 0.18) or reference_duration <= 0:
            return wav, 1.0

        ref_units = self._text_units(reference_text)
        target_units = self._text_units(target_text)
        desired_duration = reference_duration * (target_units / ref_units)
        current_duration = len(wav) / sr
        if current_duration <= 0:
            return wav, 1.0

        mixed_duration = current_duration * (1.0 - strength) + desired_duration * strength
        correction_rate = current_duration / max(mixed_duration, 1e-3)
        correction_rate = float(np.clip(correction_rate, min_rate, max_rate))
        if abs(correction_rate - 1.0) < 0.025:
            return wav, correction_rate

        stretched = librosa.effects.time_stretch(wav.astype(np.float32), rate=correction_rate)
        return np.asarray(stretched, dtype=np.float32), correction_rate

    def _generate_one(self, target_text, prompt_text, reference_path, cfg, timesteps,
                      seed, retry_badcase, retry_max_times, retry_ratio_threshold):
        model = model_manager.voxcpm
        signature = inspect.signature(model.generate)
        supported = set(signature.parameters.keys())

        kwargs = {
            "text": target_text,
            "prompt_wav_path": str(reference_path),
            "prompt_text": prompt_text,
            "reference_wav_path": str(reference_path),
            "cfg_value": float(cfg),
            "inference_timesteps": int(timesteps),
            "retry_badcase": bool(retry_badcase),
            "retry_badcase_max_times": int(retry_max_times),
            "retry_badcase_ratio_threshold": float(retry_ratio_threshold),
        }

        if "seed" in supported:
            kwargs["seed"] = int(seed)
        else:
            import torch
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        if "kwargs" not in signature.parameters:
            kwargs = {k: v for k, v in kwargs.items() if k in supported}

        if "text" not in kwargs:
            raise RuntimeError("CRITICAL: text was removed before model.generate().")

        wav = model.generate(**kwargs)
        if wav is None:
            raise RuntimeError("VoxCPM2 returned None.")
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            raise RuntimeError("VoxCPM2 returned empty audio.")
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        return normalize_peak(wav)

    async def clone(self, reference_audio, prompt_text, target_text, **options):
        if not model_manager.ready:
            raise RuntimeError("AI models are not ready.")

        raw = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}_raw"
        clean = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}_reference.wav"
        temp_candidates = []
        try:
            data = await reference_audio.read(settings.max_file_size + 1)
            if not data:
                raise ValueError("Reference audio is empty.")
            if len(data) > settings.max_file_size:
                raise ValueError(
                    f"Reference audio exceeds the maximum allowed size "
                    f"({settings.MAX_FILE_SIZE_MB} MB)."
                )
            raw.write_bytes(data)

            await asyncio.to_thread(
                prepare_reference, raw, clean, settings.REFERENCE_SAMPLE_RATE
            )
            qc = await asyncio.to_thread(reference_quality_report, clean)

            if options.get("auto_transcribe"):
                from app.services.transcription.transcription_service import TranscriptionService
                prompt_text = await TranscriptionService().transcribe_file(clean)

            if not prompt_text or not prompt_text.strip():
                raise ValueError("prompt_text is required unless auto_transcribe=true")
            if not target_text or not target_text.strip():
                raise ValueError("target_text is required")

            target_text = normalize_iraqi_target(target_text)
            base_seed = int(options.get("seed") if options.get("seed") is not None else 12345)

            # Project policy: exactly one candidate.
            count = 1

            ranked = []
            model_sr = int(getattr(
                model_manager.voxcpm.tts_model,
                "sample_rate",
                settings.OUTPUT_SAMPLE_RATE,
            ))
            chunk_long_text = bool(options.get("chunk_long_text", False))
            silence_ms = int(options.get("silence_ms", 350))
            single_pass = bool(options.get("single_pass", True))
            auto_sentence_points = bool(options.get("auto_sentence_points", True))
            enable_reference_rhythm = bool(options.get("enable_reference_rhythm", True))
            rhythm_strength = float(np.clip(
                options.get("rhythm_strength", 1.0), 0.0, 1.0
            ))

            if single_pass:
                final_target = (
                    self._auto_add_sentence_points(target_text)
                    if auto_sentence_points else target_text
                )
                chunks = [final_target]
            elif chunk_long_text:
                chunks = split_text_at_question_marks(target_text) or [target_text]
            else:
                chunks = [target_text]

            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                "Generation mode=%s, candidates=%d",
                "single-pass" if single_pass else "chunked", count
            )

            # Serialize heavy GPU work. The actual blocking calls run in worker
            # threads so FastAPI can still serve lightweight endpoints.
            async with model_manager.model_semaphore:
                for index in range(count):
                    candidate_seed = base_seed + index * 1009

                    if single_pass:
                        wav = await asyncio.to_thread(
                            self._generate_one,
                            chunks[0], prompt_text.strip(), clean,
                            options.get("cfg_value", settings.DEFAULT_CFG),
                            options.get("inference_timesteps", settings.DEFAULT_TIMESTEPS),
                            candidate_seed,
                            options.get("retry_badcase", settings.DEFAULT_RETRY_BADCASE),
                            options.get("retry_max_times", settings.DEFAULT_RETRY_MAX_TIMES),
                            options.get("retry_ratio_threshold", settings.DEFAULT_RETRY_RATIO_THRESHOLD),
                        )

                        if enable_reference_rhythm and rhythm_strength > 0:
                            try:
                                import librosa
                                ref_audio, ref_sr = await asyncio.to_thread(
                                    librosa.load, str(clean), sr=model_sr, mono=True
                                )
                                ref_duration = len(ref_audio) / float(ref_sr) if len(ref_audio) else 0.0
                                wav, correction_rate = await asyncio.to_thread(
                                    self._fit_audio_to_reference_rate,
                                    wav, ref_duration, prompt_text.strip(), chunks[0],
                                    rhythm_strength,
                                )
                                logger.info(
                                    "Reference rhythm correction: %.3fx",
                                    correction_rate
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Reference rhythm correction skipped: %s", exc
                                )
                    else:
                        segment_wavs = []
                        for seg_index, chunk in enumerate(chunks):
                            seg_seed = candidate_seed + seg_index
                            segment_wavs.append(await asyncio.to_thread(
                                self._generate_one,
                                chunk, prompt_text.strip(), clean,
                                options.get("cfg_value", settings.DEFAULT_CFG),
                                options.get("inference_timesteps", settings.DEFAULT_TIMESTEPS),
                                seg_seed,
                                options.get("retry_badcase", settings.DEFAULT_RETRY_BADCASE),
                                options.get("retry_max_times", settings.DEFAULT_RETRY_MAX_TIMES),
                                options.get("retry_ratio_threshold", settings.DEFAULT_RETRY_RATIO_THRESHOLD),
                            ))
                        wav = concatenate_with_silence(
                            segment_wavs, model_sr, silence_ms=silence_ms
                        )

                    wav = normalize_peak(wav)
                    candidate_wav = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}.wav"
                    await asyncio.to_thread(
                        sf.write, candidate_wav, wav, model_sr, subtype="PCM_24"
                    )
                    temp_candidates.append(candidate_wav)

                    similarity = self._speaker_score(clean, candidate_wav)
                    score = similarity["score"]
                    ranked.append({
                        "path": candidate_wav,
                        "score": score,
                        "rank_score": float(score) if score is not None else 0.0,
                        "same_speaker": similarity["same_speaker"],
                        "seed": candidate_seed,
                        "duration": len(wav) / model_sr,
                    })

            best = max(ranked, key=lambda x: x["rank_score"])
            output = Path(settings.OUTPUT_DIR) / f"{uuid.uuid4().hex}.ogg"
            await asyncio.to_thread(
                convert_to_ogg,
                best["path"], output,
                settings.OUTPUT_BITRATE, settings.OUTPUT_SAMPLE_RATE
            )

            return {
                "request_id": output.stem,
                "output": output,
                "prompt_text": prompt_text.strip(),
                "target_text": target_text,
                "seed": best["seed"],
                "duration": best["duration"],
                "speaker_score": best["score"],
                "same_speaker": best["same_speaker"],
                "reference_qc": qc,
                "chunks_count": len(chunks),
                "single_pass": single_pass,
                "auto_sentence_points": auto_sentence_points,
                "enable_reference_rhythm": enable_reference_rhythm,
                "rhythm_strength": rhythm_strength,
                "final_target_text": chunks[0] if single_pass else target_text,
                "ranking": [{
                    "rank": 1,
                    "seed": best["seed"],
                    "score": best["score"],
                    "same_speaker": best["same_speaker"],
                    "duration": best["duration"],
                }],
            }
        finally:
            raw.unlink(missing_ok=True)
            clean.unlink(missing_ok=True)
            for f in temp_candidates:
                Path(f).unlink(missing_ok=True)
