"""
backend/app/schemas/debug.py
Data models for debugging request/response contracts.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class DebugRequest(BaseModel):
    code: str = Field(..., description="Faulty source code")
    language: str = Field(default="python", description="Programming language")
    error_message: Optional[str] = Field(None, description="Runtime error or compiler message")
    stack_trace: Optional[str] = Field(None, description="Full traceback string if available")


class DebugResponse(BaseModel):
    error_type: str
    error_location: Optional[str]
    root_cause: str
    suggested_fix: str
    syntax_valid: bool
    confidence_score: float