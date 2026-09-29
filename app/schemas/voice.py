from pydantic import BaseModel

class VoiceCloneResponse(BaseModel):
    success: bool
    request_id: str
    format: str
    mime_type: str
    audio_url: str
    prompt_text: str | None = None
    seed: int | None = None
    duration: float | None = None
    speaker_score: float | None = None
    same_speaker: bool | None = None
    reference_qc: str | None = None
    ranking: list[dict] = []
    chunks_count: int | None = None
    single_pass: bool | None = None
    auto_sentence_points: bool | None = None
    enable_reference_rhythm: bool | None = None
    rhythm_strength: float | None = None
    final_target_text: str | None = None
