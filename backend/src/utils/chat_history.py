"""
Simple Chat History Model
Store user queries and responses in MongoDB
"""

from datetime import datetime
from typing import Optional
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class ChatHistory:
    """Simple chat history manager"""
    
    def __init__(self):
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.MONGODB_DB_NAME]
        self.collection = self.db['chat_history']
        
        # Create index on user_email and timestamp
        self.collection.create_index([("user_email", 1), ("timestamp", DESCENDING)])
    
    def save_message(
        self,
        user_email: str,
        query: str,
        response: str,
        query_type: Optional[str] = None,
        pipeline: Optional[str] = None
    ) -> str:
        """Save a chat message"""
        message = {
            "user_email": user_email,
            "query": query,
            "response": response,
            "query_type": query_type,
            "pipeline": pipeline,
            "timestamp": datetime.now()
        }
        
        result = self.collection.insert_one(message)
        return str(result.inserted_id)
    
    def get_user_history(self, user_email: str, limit: int = 50):
        """Get user's chat history (newest first, then reversed)"""
        messages = self.collection.find(
            {"user_email": user_email}
        ).sort("timestamp", DESCENDING).limit(limit)
        
        history = []
        for msg in messages:
            history.append({
                "id": str(msg["_id"]),
                "query": msg["query"],
                "response": msg["response"],
                "query_type": msg.get("query_type"),
                "pipeline": msg.get("pipeline"),
                "timestamp": msg["timestamp"].isoformat()
            })
        
        # Reverse to show oldest first
        return list(history)
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()


# Singleton instance
_chat_history_instance = None

def get_chat_history() -> ChatHistory:
    """Get singleton chat history instance"""
    global _chat_history_instance
    
    if _chat_history_instance is None:
        _chat_history_instance = ChatHistory()
    
    return _chat_history_instance