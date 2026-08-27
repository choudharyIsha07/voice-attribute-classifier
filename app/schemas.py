from pydantic import BaseModel
from typing import Literal, Optional
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
    language: Optional[str] = None   # e.g. "en", "es", "unknown" — best-effort bonus field


class WebSocketChunkResult(BaseModel):
    """Progressive result emitted over WebSocket for each chunk."""
    chunk_index: int
    gender: Prediction
    age_bracket: Prediction
    audio_quality: Literal["good", "degraded", "insufficient"]
    language: Optional[str] = None
    is_final: bool = False
