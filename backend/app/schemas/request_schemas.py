from pydantic import BaseModel
from typing import Optional, List

class SignEvaluationRequest(BaseModel):
    """Request schema for sign evaluation."""
    expected_sign: Optional[str] = None

class SignDescriptionRequest(BaseModel):
    """Request schema for sign description."""
    word: str

class SignEvaluationResponse(BaseModel):
    """Response schema for sign evaluation."""
    predicted_sign: str
    confidence: float
    feedback: str
    is_correct: bool

class SignDescriptionResponse(BaseModel):
    """Response schema for sign description."""
    word: str
    description: str
    steps: List[str]
    tips: List[str]

class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str 