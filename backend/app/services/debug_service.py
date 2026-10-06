"""
backend/app/services/debug_service.py
Coordinates code analysis, context assembly, and local LLM inference
with robust error boundaries.
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.config import settings
from backend.app.debugging.analyzer import CodeAnalyzer
from backend.app.schemas.debug import DebugRequest, DebugResponse
from llm.inference.generate import DebugGenerator


class DebugService:
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.generator = None

    def _ensure_generator_loaded(self):
        if self.generator is None:
            if not os.path.exists(settings.MODEL_CHECKPOINT):
                raise FileNotFoundError(
                    f"Model checkpoint not found at {settings.MODEL_CHECKPOINT}."
                )
            self.generator = DebugGenerator(
                model_path=settings.MODEL_CHECKPOINT,
                bpe_dir=settings.BPE_DIR
            )

    def analyze_and_repair(self, request: DebugRequest) -> DebugResponse:
        # Step 1: Deterministic Syntax Check
        is_syntax_valid, syntax_err, line_no = self.analyzer.inspect_syntax(request.code)

        # Step 2: Context Construction
        error_info = request.error_message or (
            f"SyntaxError at line {line_no}: {syntax_err}" if not is_syntax_valid else "Unknown Error"
        )
        prompt = f"<|code|>\n{request.code}\n<|error|>\n{error_info}\n<|fix|>\n"

        # Step 3: Run Model Generation
        self._ensure_generator_loaded()
        try:
            generated_fix = self.generator.generate(
                prompt=prompt,
                max_new_tokens=90,
                temperature=0.4,
                top_k=8,
                repetition_penalty=1.35
            ).strip()
        except Exception as gen_err:
            generated_fix = f"# Generation failed: {str(gen_err)}"

        # Step 4: Validate Fix Syntax
        fix_syntax_valid, _, _ = self.analyzer.inspect_syntax(generated_fix)

        # Step 5: Format Response
        err_type = error_info.split(":")[0].strip() if ":" in error_info else "RuntimeError"

        return DebugResponse(
            error_type=err_type,
            error_location=f"Line {line_no}" if line_no else "Function Scope",
            root_cause=error_info,
            suggested_fix=generated_fix if generated_fix else request.code,
            syntax_valid=bool(fix_syntax_valid),
            confidence_score=0.85 if fix_syntax_valid else 0.45
        )


debug_service = DebugService()