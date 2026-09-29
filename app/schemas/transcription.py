from pydantic import BaseModel

class TranscriptionResponse(BaseModel):
    success: bool
    request_id: str
    text: str
    language: str
