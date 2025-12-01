from datetime import datetime
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
import sys
from pathlib import Path
import json
import logging

# Add parent directory to path to import chatbot
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from .schemas import ChatRequest, HealthResponse, StatusChunk
from hr_chatbot import HRChatbot

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize chatbot once (singleton pattern)
try:
    chatbot = HRChatbot()
    logger.info("✅ Chatbot initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize chatbot: {e}")
    chatbot = None

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE)
    
    Yields chunks in format: data: {json}\n\n
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot service unavailable")
    
    # Add this custom encoder at the top of routes.py
    class SafeJSONEncoder(json.JSONEncoder):
        """Custom JSON encoder that handles MongoDB ObjectId and datetime"""
        def default(self, obj):
            if isinstance(obj, ObjectId):
                return str(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
    
    async def generate():
        try:
             # Stream from chatbot
            for chunk in chatbot.ask_stream(
                query=request.query,
                auto_confirm_ultra=request.auto_confirm_ultra
            ):
                # Convert chunk to dict if it's a Pydantic model
                if hasattr(chunk, 'model_dump'):
                    chunk_dict = chunk.model_dump()
                elif hasattr(chunk, 'dict'):
                    chunk_dict = chunk.dict()
                elif isinstance(chunk, dict):
                    chunk_dict = chunk
                else:
                    logger.warning(f"Unexpected chunk type: {type(chunk)}")
                    chunk_dict = {
                        'type': 'error',
                        'message': f'Unexpected data type: {type(chunk).__name__}',
                        'raw_value': str(chunk)
                    }
                
                # Format as SSE with custom encoder
                yield f"data: {json.dumps(chunk_dict, cls=SafeJSONEncoder)}\n\n"
                
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_chunk = {
                'type': 'error',
                'message': 'An error occurred during streaming',
                'details': str(e)
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    # Return streaming response
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        stats = chatbot.get_stats()
        return HealthResponse(status="healthy", stats=stats)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")