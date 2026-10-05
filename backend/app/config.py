"""
backend/app/config.py
Application settings and environment configurations.
"""
import os
from pydantic import BaseModel

CURRENT_FILE = os.path.abspath(__file__)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(CURRENT_FILE), "..", ".."))


class Settings(BaseModel):
    PROJECT_NAME: str = "TracePilot API"
    VERSION: str = "0.1.0"
    MODEL_CHECKPOINT: str = os.path.join(ROOT_DIR, "llm", "checkpoints", "best_debugger.pt")
    BPE_DIR: str = os.path.join(ROOT_DIR, "llm", "tokenizer", "bpe_model")
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # MongoDB Configuration
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
    DATABASE_NAME: str = "tracepilot_db"


settings = Settings()