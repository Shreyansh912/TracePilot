"""
backend/app/schemas/debug.py
Data validation schemas for debug requests and multi-pass sandbox responses.
"""
from typing import Optional, List
from pydantic import BaseModel


class DebugRequest(BaseModel):
    code: str
    language: str = "python"
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None


class IterationAttempt(BaseModel):
    attempt_number: int
    patch_code: str
    passed: bool
    error_summary: str


class DebugResponse(BaseModel):
    error_type: str
    error_location: str
    root_cause: str
    suggested_fix: str
    syntax_valid: bool
    confidence_score: float
    sandbox_executed: bool = False
    sandbox_passed: bool = False
    sandbox_stdout: str = ""
    sandbox_stderr: str = ""
    # Self-Correction Loop Telemetry
    iterations_used: int = 1
    self_corrected: bool = False
    attempts: List[IterationAttempt] = []