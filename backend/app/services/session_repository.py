"""
backend/app/services/session_repository.py
Repository handling asynchronous storage of debugging sessions,
model outputs, and verification metrics in MongoDB.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from bson import ObjectId
from backend.app.database import get_database


class SessionRepository:
    @staticmethod
    async def create_session(
        code: str,
        error_message: Optional[str],
        stack_trace: Optional[str],
        diagnosis: Dict[str, Any],
        project_id: Optional[str] = None
    ) -> Optional[str]:
        db = get_database()
        if db is None:
            return None  # MongoDB not connected

        record = {
            "project_id": project_id or "default",
            "code_input": code,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "diagnosis": diagnosis,
            "model_version": "minigpt-debugger-v1",
            "created_at": datetime.utcnow()
        }

        result = await db["debug_sessions"].insert_one(record)
        return str(result.inserted_id)

    @staticmethod
    async def get_recent_sessions(limit: int = 10) -> List[Dict[str, Any]]:
        db = get_database()
        if db is None:
            return []

        cursor = db["debug_sessions"].find().sort("created_at", -1).limit(limit)
        sessions = []
        async for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            sessions.append(doc)
        return sessions