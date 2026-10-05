"""
backend/app/config.py
Application settings and robust path configurations.
"""
import os
from pydantic import BaseModel

# Dynamically locate the repository root (the folder containing llm/ and datasets/)
CURRENT_FILE = os.path.abspath(__file__)
# __file__ is in backend/app/config.py -> go up 2 levels to get to repository root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(CURRENT_FILE), "..", ".."))


class Settings(BaseModel):
    PROJECT_NAME: str = "TracePilot API"
    VERSION: str = "0.1.0"
    MODEL_CHECKPOINT: str = os.path.join(ROOT_DIR, "llm", "checkpoints", "best_debugger.pt")
    BPE_DIR: str = os.path.join(ROOT_DIR, "llm", "tokenizer", "bpe_model")
    HOST: str = "127.0.0.1"
    PORT: int = 8000


settings = Settings()