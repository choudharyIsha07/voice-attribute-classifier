from pydantic import BaseModel
from typing import Literal
from uuid import UUID

class Prediction(BaseModel):
    prediction: str
    confidence: float

class AnalyzeResponse(BaseModel):
    contact_id: UUID
    gender: Prediction
    age_bracket: Prediction
    processing_ms: int
    audio_quality: Literal["good", "degraded", "insufficient"]
