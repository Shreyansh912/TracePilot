"""
backend/app/main.py
TracePilot FastAPI Application entry point with MongoDB lifecycle hooks.
"""
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
    # Startup: Initialize DB Connection pool
    await connect_to_mongo()
    yield
    # Shutdown: Close pool
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
        
        # Persist debug session asynchronously to MongoDB
        await SessionRepository.create_session(
            code=request.code,
            error_message=request.error_message,
            stack_trace=request.stack_trace,
            diagnosis=response.dict()
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/history")
async def get_history(limit: int = 10):
    return await SessionRepository.get_recent_sessions(limit=limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)