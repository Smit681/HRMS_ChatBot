from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
from enum import Enum

from .auth_schemas import UserRegister, UserLogin, Token, UserResponse

class ChunkType(str, Enum):
    """Types of streaming chunks"""
    STATUS = "status"
    CLASSIFICATION = "classification"
    TOKEN = "token"
    METADATA = "metadata"
    COMPLETE = "complete"
    ERROR = "error"
    CONFIRMATION_REQUIRED = "confirmation_required"


class ChatRequest(BaseModel):
    """Request schema for chat endpoint"""
    query: str = Field(..., min_length=1, max_length=1000, description="User's question")
    auto_confirm_ultra: bool = Field(default=False, description="Auto-confirm ultra-complex queries")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How many employees have H-1B visas?",
                "auto_confirm_ultra": False
            }
        }


class StatusChunk(BaseModel):
    """Status update during processing"""
    type: Literal[ChunkType.STATUS] = ChunkType.STATUS
    message: str


class ClassificationChunk(BaseModel):
    """Query classification result"""
    type: Literal[ChunkType.CLASSIFICATION] = ChunkType.CLASSIFICATION
    query_type: str
    confidence: float


class TokenChunk(BaseModel):
    """Single token from LLM"""
    type: Literal[ChunkType.TOKEN] = ChunkType.TOKEN
    content: str


class MetadataChunk(BaseModel):
    """Metadata after completion"""
    type: Literal[ChunkType.METADATA] = ChunkType.METADATA
    sources: Optional[List[Dict[str, Any]]] = None
    num_sources: Optional[int] = None
    confidence: Optional[float] = None
    pipeline: Optional[str] = None
    mongodb_operation: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    total_analyzed: Optional[int] = None


class CompleteChunk(BaseModel):
    """Final completion message"""
    type: Literal[ChunkType.COMPLETE] = ChunkType.COMPLETE
    query_type: str
    processing_time: float


class ErrorChunk(BaseModel):
    """Error message"""
    type: Literal[ChunkType.ERROR] = ChunkType.ERROR
    message: str
    details: Optional[str] = None


class ConfirmationChunk(BaseModel):
    """Requires user confirmation"""
    type: Literal[ChunkType.CONFIRMATION_REQUIRED] = ChunkType.CONFIRMATION_REQUIRED
    message: str
    query_type: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    stats: Dict[str, Any]