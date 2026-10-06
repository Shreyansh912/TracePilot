"""
backend/app/services/debug_service.py
Coordinates AST scope isolation, traceback symbol matching,
local MiniGPT inference, and isolated sandbox verification.
"""
import os
import sys
import ast
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.config import settings
from backend.app.debugging.analyzer import CodeAnalyzer
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
        # Step 1: Clean and extract specific error line
        raw_error = request.error_message or ""
        if not raw_error and request.stack_trace:
            lines = [l.strip() for l in request.stack_trace.strip().split("\n") if l.strip()]
            raw_error = lines[-1] if lines else "RuntimeError"
        clean_error = raw_error.split("\n")[-1].strip()

        # Step 2: Extract target function symbol from stack trace
        target_func_name = None
        match = re.search(r"in\s+([a-zA-Z_0-9]+)", request.stack_trace or "")
        if match:
            target_func_name = match.group(1)

        func_def_code = request.code.strip()
        non_target_nodes = []

        try:
            tree = ast.parse(request.code)
            matched_func = None

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if target_func_name and node.name == target_func_name:
                        matched_func = ast.unparse(node)
                    else:
                        non_target_nodes.append(ast.unparse(node))
                else:
                    non_target_nodes.append(ast.unparse(node))

            if matched_func:
                func_def_code = matched_func
            else:
                for i, node in enumerate(tree.body):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_def_code = ast.unparse(node)
                        non_target_nodes = [
                            ast.unparse(n) for j, n in enumerate(tree.body) if j != i
                        ]
                        break
        except Exception:
            pass

        # Step 3: Exact Training Prompt
        prompt = f"<|code|>\n{func_def_code}\n<|error|>\n{clean_error}\n<|fix|>\n"
        print(f"\n[TracePilot Prompt]:\n{prompt}")

        # Step 4: Run Inference via MiniGPT
        self._ensure_generator_loaded()
        try:
            raw_generation = self.generator.generate(
                prompt=prompt,
                max_new_tokens=90,
                min_new_tokens=12,
                temperature=0.25,
                top_k=5,
                repetition_penalty=1.35
            ).strip()

            print(f"[TracePilot Raw Output]:\n{raw_generation}\n")

            cleaned_fix = raw_generation.replace("<|eos|>", "").strip()

            # Splice repaired function with original driver/caller code
            if cleaned_fix and "def " in cleaned_fix:
                if non_target_nodes:
                    suggested_fix = f"{cleaned_fix}\n\n" + "\n".join(non_target_nodes)
                else:
                    suggested_fix = cleaned_fix
            else:
                suggested_fix = cleaned_fix if cleaned_fix else request.code

        except Exception as gen_err:
            print(f"[!] Generation error: {gen_err}")
            suggested_fix = request.code

        # Step 5: Verify AST Syntax
        is_syntax_valid, _, _ = self.analyzer.inspect_syntax(suggested_fix)

        # Step 6: Isolated Subprocess Execution Verification
        sandbox_res = self.sandbox.execute(suggested_fix)

        if isinstance(sandbox_res, dict):
            s_passed = bool(sandbox_res.get("success", False))
            s_stdout = str(sandbox_res.get("stdout", ""))
            s_stderr = str(sandbox_res.get("stderr", ""))
        else:
            s_passed = bool(getattr(sandbox_res, "success", False))
            s_stdout = str(getattr(sandbox_res, "stdout", ""))
            s_stderr = str(getattr(sandbox_res, "stderr", ""))

        confidence = 0.50
        if is_syntax_valid:
            confidence += 0.25
        if s_passed:
            confidence += 0.25

        err_type = clean_error.split(":")[0].strip() if ":" in clean_error else "RuntimeError"

        return DebugResponse(
            error_type=err_type,
            error_location=f"Function: {target_func_name}" if target_func_name else "Function Scope",
            root_cause=clean_error,
            suggested_fix=suggested_fix,
            syntax_valid=bool(is_syntax_valid),
            confidence_score=round(confidence, 2),
            sandbox_executed=True,
            sandbox_passed=s_passed,
            sandbox_stdout=s_stdout,
            sandbox_stderr=s_stderr
        )


debug_service = DebugService()