"""
backend/app/schemas/debug.py
Data validation schemas for debug requests and sandbox-verified responses.
"""
from typing import Optional
from pydantic import BaseModel


class DebugRequest(BaseModel):
    code: str
    language: str = "python"
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None


class DebugResponse(BaseModel):
    error_type: str
    error_location: str
    root_cause: str
    suggested_fix: str
    syntax_valid: bool
    confidence_score: float
    # Sandbox execution telemetry
    sandbox_executed: bool = False
    sandbox_passed: bool = False
    sandbox_stdout: str = ""
    sandbox_stderr: str = ""