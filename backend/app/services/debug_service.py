"""
backend/app/services/debug_service.py
Coordinates AST scope isolation, local LLM inference, and isolated sandbox verification.
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.config import settings
from backend.app.debugging.analyzer import CodeAnalyzer
from backend.app.debugging.context_builder import ContextBuilder
from backend.app.sandbox.runner import CodeSandbox
from backend.app.schemas.debug import DebugRequest, DebugResponse
from llm.inference.generate import DebugGenerator


class DebugService:
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.sandbox = CodeSandbox(timeout_seconds=2.0)
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
        # 1. AST Scope Isolation & Prompt Assembly
        assembled = ContextBuilder.assemble_prompt(
            code=request.code,
            raw_trace=request.stack_trace,
            error_message=request.error_message
        )

        prompt = assembled["prompt"]
        target_line = assembled.get("target_line")
        error_type = assembled.get("error_type", "RuntimeError")
        error_msg = assembled.get("error_message", "Unknown failure")

        # 2. Local Model Inference
        self._ensure_generator_loaded()
        try:
            raw_generation = self.generator.generate(
                prompt=prompt,
                max_new_tokens=70,
                temperature=0.3,
                top_k=5,
                repetition_penalty=1.45
            ).strip()

            cleaned_fix = raw_generation.replace("<|eos|>", "").strip()
            lines = [l for l in cleaned_fix.split("\n") if l.strip()]
            suggested_fix = "\n".join(lines) if lines else request.code
        except Exception as gen_err:
            suggested_fix = f"# Generation notice: {str(gen_err)}\n{assembled['focused_code']}"

        # 3. Static AST Syntax Check
        is_syntax_valid, _, _ = self.analyzer.inspect_syntax(suggested_fix)

        # 4. Isolated Subprocess Sandbox Verification
        sandbox_res = self.sandbox.execute(suggested_fix)

        # 5. Composite Confidence Scoring
        confidence = 0.50
        if is_syntax_valid:
            confidence += 0.25
        if sandbox_res.success:
            confidence += 0.20

        return DebugResponse(
            error_type=error_type,
            error_location=f"Line {target_line}" if target_line else "Function Scope",
            root_cause=f"{error_type}: {error_msg}",
            suggested_fix=suggested_fix,
            syntax_valid=bool(is_syntax_valid),
            confidence_score=round(confidence, 2),
            sandbox_executed=True,
            sandbox_passed=bool(sandbox_res.success),
            sandbox_stdout=sandbox_res.stdout or "",
            sandbox_stderr=sandbox_res.stderr or ""
        )


debug_service = DebugService()