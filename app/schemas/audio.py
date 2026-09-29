from pydantic import BaseModel

class MergeResponse(BaseModel):
    success: bool
    request_id: str
    format: str
    mime_type: str
    audio_url: str
    input_count: int
