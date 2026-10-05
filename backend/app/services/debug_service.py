"""
backend/app/services/debug_service.py
Coordinates code analysis, context assembly, and local LLM inference.
"""
import os
import sys
from typing import Dict, Any

# Ensure root is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from backend.app.config import settings
from backend.app.debugging.analyzer import CodeAnalyzer
from backend.app.schemas.debug import DebugRequest, DebugResponse
from llm.inference.generate import DebugGenerator


class DebugService:
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        # Lazily loaded generator instance
        self.generator = None

    def _ensure_generator_loaded(self):
        if self.generator is None:
            if not os.path.exists(settings.MODEL_CHECKPOINT):
                raise FileNotFoundError(
                    f"Model checkpoint not found at {settings.MODEL_CHECKPOINT}. Train model first."
                )
            self.generator = DebugGenerator(
                model_path=settings.MODEL_CHECKPOINT,
                bpe_dir=settings.BPE_DIR
            )

    def analyze_and_repair(self, request: DebugRequest) -> DebugResponse:
        # Step 1: Programmatic Static Analysis
        is_syntax_valid, syntax_err, line_no = self.analyzer.inspect_syntax(request.code)
        
        # Step 2: Build Context Prompt for Local LLM
        error_info = request.error_message or (f"SyntaxError at line {line_no}: {syntax_err}" if not is_syntax_valid else "Unknown Error")
        prompt = f"<|code|>\n{request.code}\n<|error|>\n{error_info}\n<|fix|>\n"

        # Step 3: Run Inference through our Local Transformer
        self._ensure_generator_loaded()
        generated_fix = self.generator.generate(
            prompt=prompt,
            max_new_tokens=90,
            temperature=0.4,
            top_k=8,
            repetition_penalty=1.35
        )

        # Step 4: Validate generated patch syntax
        fix_syntax_valid, _, _ = self.analyzer.inspect_syntax(generated_fix.strip())

        # Step 5: Classify error type deterministically or fallback to general runtime
        err_type = error_info.split(":")[0].strip() if ":" in error_info else "RuntimeError"

        return DebugResponse(
            error_type=err_type,
            error_location=f"Line {line_no}" if line_no else "Function Scope",
            root_cause=error_info,
            suggested_fix=generated_fix.strip(),
            syntax_valid=fix_syntax_valid,
            confidence_score=0.85 if fix_syntax_valid else 0.45
        )


debug_service = DebugService()