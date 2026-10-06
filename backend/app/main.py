"""
backend/app/main.py
TracePilot FastAPI Application entry point with resilient error handling.
"""
import sys
import os

# Insert workspace root into sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.database import connect_to_mongo, close_mongo_connection
from backend.app.schemas.debug import DebugRequest, DebugResponse
from backend.app.services.debug_service import debug_service
from backend.app.services.session_repository import SessionRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield
    await close_mongo_connection()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Autonomous AI Debugging Assistant powered by custom small Transformer",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }


@app.post("/debug", response_model=DebugResponse)
async def debug_code(request: DebugRequest):
    try:
        response = debug_service.analyze_and_repair(request)
        
        # Serialize response safely across Pydantic v1 and v2
        diag_dict = response.model_dump() if hasattr(response, "model_dump") else response.dict()

        # Non-blocking MongoDB persistence with failure isolation
        try:
            await SessionRepository.create_session(
                code=request.code,
                error_message=request.error_message,
                stack_trace=request.stack_trace,
                diagnosis=diag_dict
            )
        except Exception as db_err:
            print(f"[!] Database persistence skipped: {db_err}")

        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/history")
async def get_history(limit: int = 10):
    return await SessionRepository.get_recent_sessions(limit=limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)