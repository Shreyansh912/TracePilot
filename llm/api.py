import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from llm.engine import TracePilotEngine

engine: Optional[TracePilotEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Startup: Load the model onto GPU
    engine = TracePilotEngine()
    yield
    # Shutdown logic if needed
    engine = None


app = FastAPI(title="TracePilot AI Code Repair API", version="1.0.0", lifespan=lifespan)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RepairRequest(BaseModel):
    code: str
    error: str
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 64


class RepairResponse(BaseModel):
    patch: str
    ast_valid: bool
    tokens_generated: int
    latency_ms: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(engine.device) if engine else "uninitialized",
        "model": "TracePilot MiniGPT",
    }


@app.post("/repair", response_model=RepairResponse)
def repair_code(payload: RepairRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Model engine not loaded.")
    if not payload.code.strip():
        raise HTTPException(status_code=400, detail="Buggy code cannot be empty.")

    result = engine.generate_patch(
        buggy_code=payload.code,
        error_message=payload.error,
        temperature=payload.temperature,
        max_new_tokens=payload.max_tokens,
    )
    return result


if __name__ == "__main__":
    import uvicorn
    # Use port 8000 (or switch to 8001 if another app permanently needs 8000)
    uvicorn.run("llm.api:app", host="127.0.0.1", port=8000, reload=False)