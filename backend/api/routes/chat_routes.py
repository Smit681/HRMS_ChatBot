import asyncio
from datetime import datetime
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Response
from fastapi.params import Depends
from fastapi.responses import StreamingResponse
import sys
from pathlib import Path
import json
import logging


from ..routes.auth_routes import get_current_user

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "backend" / "src"))
from utils.entity_tracker import get_entity_tracker
entity_tracker = get_entity_tracker()

# Add parent directory to path to import chatbot
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from ..schemas import ChatRequest, HealthResponse, StatusChunk
from hr_chatbot import HRChatbot

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from utils.chat_history import get_chat_history

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize chatbot once (singleton pattern)
try:
    chatbot = HRChatbot()
    logger.info("✅ Chatbot initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize chatbot: {e}")
    chatbot = None

# Initialize chat history
chat_history = get_chat_history()

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Streaming chat endpoint using Server-Sent Events (SSE)
    
    Yields chunks in format: data: {json}\n\n
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot service unavailable")
    
    user_email = current_user["email"]

    recent_conversation = chat_history.get_recent_conversation(user_email, limit=2)
    entity_tracker.update_entity(user_email, recent_conversation[0]["query"] if recent_conversation else "", recent_conversation[0]["response"] if recent_conversation else "")

    
    original_query = request.query
    resolved_query = entity_tracker.resolve_query(user_email, original_query)

    print("\n\n\n\n\n\n\n\nResolved Query:", resolved_query)

    if resolved_query != original_query:
        logger.info(f"📝 Query resolved: '{original_query}' → '{resolved_query}'")

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
        full_response = ""
        query_type = None
        pipeline = None
        
        try:
            # Stream from chatbot
            async for chunk in chatbot.ask_stream(
                query= resolved_query,
                auto_confirm_ultra=request.auto_confirm_ultra,
                conversation_history=recent_conversation
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
                
                # Capture metadata for history
                if chunk_dict.get('type') == 'classification':
                    query_type = chunk_dict.get('query_type')
                
                if chunk_dict.get('type') == 'token':
                    full_response += chunk_dict.get('content', '')
                
                if chunk_dict.get('type') == 'metadata':
                    pipeline = chunk_dict.get('pipeline')
                
                # Format as SSE with custom encoder
                yield f"data: {json.dumps(chunk_dict, cls=SafeJSONEncoder)}\n\n"
                await asyncio.sleep(0)  # Yield control to event loop
            
            # Save to history after streaming completes
            if full_response:
                try:
                    chat_history.save_message(
                        user_email=user_email,
                        query=resolved_query,
                        response=full_response,
                        query_type=query_type,
                        pipeline=pipeline
                    )
                    logger.info(f"💾 Saved chat history for {user_email}")
                except Exception as e:
                    logger.error(f"Failed to save chat history: {e}")
                
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

@router.get("/chat/history")
async def get_history(
    current_user: dict = Depends(get_current_user),
    limit: int = 50
):
    """Get user's chat history"""
    try:
        limit = min(limit, 100)  # Max 100
        
        history = chat_history.get_user_history(
            user_email=current_user["email"],
            limit=limit
        )
        
        return {
            "history": history,
            "count": len(history),
            "user": current_user["email"]
        }
    
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

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
    
@router.get("/test-stream")
async def test_stream():
    import asyncio
    
    async def generate():
        for i in range(10):
            yield f"data: {json.dumps({'count': i})}\n\n"
            await asyncio.sleep(0.5)  # Simulate delay
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )