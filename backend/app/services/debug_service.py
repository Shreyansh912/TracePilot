"""
backend/app/services/debug_service.py
Coordinates AST scope isolation, context assembly, and local LLM inference.
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.config import settings
from backend.app.debugging.analyzer import CodeAnalyzer
from backend.app.debugging.context_builder import ContextBuilder
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
        # Step 1: Use ContextBuilder to parse traceback and isolate enclosing function scope
        assembled = ContextBuilder.assemble_prompt(
            code=request.code,
            raw_trace=request.stack_trace,
            error_message=request.error_message
        )

        prompt = assembled["prompt"]
        target_line = assembled.get("target_line")
        error_type = assembled.get("error_type", "RuntimeError")
        error_msg = assembled.get("error_message", "Unknown failure")

        # Step 2: Run inference with our hardened generator
        self._ensure_generator_loaded()
        try:
            raw_generation = self.generator.generate(
                prompt=prompt,
                max_new_tokens=70,
                temperature=0.3,
                top_k=5,
                repetition_penalty=1.45
            ).strip()

            # Clean output: strip out any trailing special tokens
            cleaned_fix = raw_generation.replace("<|eos|>", "").strip()

            # If generation produced pure code or explanation + code, keep valid lines
            lines = [l for l in cleaned_fix.split("\n") if l.strip()]
            suggested_fix = "\n".join(lines) if lines else request.code

        except Exception as gen_err:
            suggested_fix = f"# Generation notice: {str(gen_err)}\n{assembled['focused_code']}"

        # Step 3: Verify AST syntax of the generated code
        is_syntax_valid, syntax_err, _ = self.analyzer.inspect_syntax(suggested_fix)

        return DebugResponse(
            error_type=error_type,
            error_location=f"Line {target_line}" if target_line else "Function Scope",
            root_cause=f"{error_type}: {error_msg}",
            suggested_fix=suggested_fix,
            syntax_valid=bool(is_syntax_valid),
            confidence_score=0.88 if is_syntax_valid else 0.40
        )


debug_service = DebugService()