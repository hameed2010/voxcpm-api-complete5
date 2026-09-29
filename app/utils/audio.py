import os
import re
import subprocess
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

def ffmpeg_path():
    path = shutil_which("ffmpeg")
    if not path:
        raise RuntimeError("FFmpeg is required but was not found in PATH.")
    return path

def shutil_which(name):
    import shutil
    return shutil.which(name)

def audio_duration(path):
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)

def normalize_peak(audio, peak=0.97):
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    current_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current_peak > peak:
        audio = audio / current_peak * peak
    return audio

def prepare_reference(input_path, output_path, target_sr=16000, top_db=35):
    audio, _ = librosa.load(str(input_path), sr=target_sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        raise ValueError("The reference audio is empty.")
    audio = audio - float(np.mean(audio))
    trimmed, _ = librosa.effects.trim(
        audio, top_db=top_db, frame_length=2048, hop_length=512
    )
    if len(trimmed) == 0:
        raise ValueError("No valid speech was found in the reference.")
    trimmed = normalize_peak(trimmed)
    sf.write(str(output_path), trimmed, target_sr, subtype="PCM_16")
    return {
        "path": str(output_path),
        "sample_rate": target_sr,
        "duration": len(trimmed) / target_sr,
        "samples": len(trimmed),
    }

def reference_quality_report(path):
    duration = audio_duration(path)
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))
    report = [
        f"Duration: {duration:.2f}s",
        f"Sample rate: {sr} Hz",
        f"RMS: {rms:.6f}",
        f"Peak: {peak:.6f}",
        f"Clipping ratio: {clipping_ratio * 100:.4f}%",
    ]
    if duration < 3:
        report.append("❌ Very short reference.")
    elif duration < 8:
        report.append("⚠️ Reference is usable but short.")
    elif duration <= 25:
        report.append("✅ Reference duration is suitable.")
    else:
        report.append("⚠️ Reference is long; a cleaner shorter clip may be better.")
    report.append(
        "❌ Noticeable clipping detected."
        if clipping_ratio > 0.005
        else "✅ No significant clipping detected."
    )
    return "\n".join(report)

def convert_to_wav(input_file, output_file, sample_rate=24000, channels=1):
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_file), "-vn",
        "-ac", str(channels), "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(output_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg conversion failed")
    if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
        raise RuntimeError("FFmpeg did not create a valid WAV file.")

def convert_to_ogg(input_file, output_file, bitrate="96k", sample_rate=48000):
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_file), "-vn",
        "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "libopus", "-b:a", bitrate,
        str(output_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg OGG conversion failed")
    if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
        raise RuntimeError("OGG output was not created.")

def merge_audio_files(input_files, output_file, sample_rate=24000, channels=1):
    if len(input_files) < 2:
        raise ValueError("At least two audio files are required.")
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y"]
    for item in input_files:
        cmd += ["-i", str(item)]
    labels = "".join(f"[{i}:a]" for i in range(len(input_files)))
    filter_complex = f"{labels}concat=n={len(input_files)}:v=0:a=1[merged]"
    cmd += [
        "-filter_complex", filter_complex, "-map", "[merged]",
        "-ac", str(channels), "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(output_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg merge failed")
    if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
        raise RuntimeError("Merged audio was not created.")

IRAQI_TARGET_NORMALIZATION = {
    "أشكو": "اشكو", "أريد": "اريد", "إريد": "اريد",
    "إنت": "انت", "إنتي": "انتي", "إنتو": "انتو",
}

def normalize_iraqi_target(text):
    if not text:
        return ""
    text = str(text).strip()
    for source, target in IRAQI_TARGET_NORMALIZATION.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def split_text_at_question_marks(text):
    """Split Arabic text at ؟, keeping the mark attached to its segment."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", str(text).strip())
    parts = re.split(r"(?<=\u061f)\s*", text)
    return [p.strip() for p in parts if p.strip()]


def concatenate_with_silence(wav_list, sample_rate, silence_ms=350):
    """Concatenate audio arrays with a silence gap between each segment."""
    if not wav_list:
        raise RuntimeError("No audio segments to concatenate.")
    if len(wav_list) == 1:
        return wav_list[0]
    silence = np.zeros(int(sample_rate * silence_ms / 1000.0), dtype=np.float32)
    parts = []
    for index, wav in enumerate(wav_list):
        parts.append(np.asarray(wav, dtype=np.float32))
        if index < len(wav_list) - 1:
            parts.append(silence)
    return np.concatenate(parts)
