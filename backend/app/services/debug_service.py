"""
backend/app/services/debug_service.py
Coordinates AST scope isolation, traceback symbol matching,
local MiniGPT inference, and multi-pass iterative sandbox verification.
"""
import os
import sys
import ast
import re
from typing import Tuple, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.config import settings
from backend.app.debugging.analyzer import CodeAnalyzer
from backend.app.sandbox.runner import CodeSandbox
from backend.app.schemas.debug import DebugRequest, DebugResponse, IterationAttempt
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

    def _extract_target_function(self, code: str, stack_trace: str) -> Tuple[str, List[str], str]:
        """
        Isolates the buggy function named in the stack trace from caller/driver code.
        """
        target_func_name = None
        match = re.search(r"in\s+([a-zA-Z_0-9]+)", stack_trace or "")
        if match:
            target_func_name = match.group(1)

        func_def_code = code.strip()
        non_target_nodes = []

        try:
            tree = ast.parse(code)
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

        return func_def_code, non_target_nodes, target_func_name or "Function Scope"

    def _run_sandbox(self, code_str: str) -> Tuple[bool, str, str]:
        sandbox_res = self.sandbox.execute(code_str)
        if isinstance(sandbox_res, dict):
            return (
                bool(sandbox_res.get("success", False)),
                str(sandbox_res.get("stdout", "")),
                str(sandbox_res.get("stderr", ""))
            )
        return (
            bool(getattr(sandbox_res, "success", False)),
            str(getattr(sandbox_res, "stdout", "")),
            str(getattr(sandbox_res, "stderr", ""))
        )

    def analyze_and_repair(self, request: DebugRequest) -> DebugResponse:
        self._ensure_generator_loaded()

        # Step 1: Error Extraction
        raw_error = request.error_message or ""
        if not raw_error and request.stack_trace:
            lines = [l.strip() for l in request.stack_trace.strip().split("\n") if l.strip()]
            raw_error = lines[-1] if lines else "RuntimeError"
        current_error = raw_error.split("\n")[-1].strip()

        # Step 2: AST Function Scope Extraction
        func_def_code, non_target_nodes, location_desc = self._extract_target_function(
            request.code, request.stack_trace or ""
        )

        attempts_history: List[IterationAttempt] = []
        max_attempts = 2
        active_code_target = func_def_code
        final_suggested_fix = request.code
        final_passed = False
        final_stdout = ""
        final_stderr = ""
        is_syntax_valid = False

        # Step 3: Multi-Pass Iterative Repair Loop
        for attempt in range(1, max_attempts + 1):
            prompt = f"<|code|>\n{active_code_target}\n<|error|>\n{current_error}\n<|fix|>\n"

            try:
                raw_generation = self.generator.generate(
                    prompt=prompt,
                    max_new_tokens=90,
                    min_new_tokens=12,
                    temperature=0.2 if attempt == 1 else 0.4,
                    top_k=5,
                    repetition_penalty=1.35
                ).strip()

                cleaned_fix = raw_generation.replace("<|eos|>", "").strip()

                if cleaned_fix and "def " in cleaned_fix:
                    candidate_full_code = (
                        f"{cleaned_fix}\n\n" + "\n".join(non_target_nodes)
                        if non_target_nodes else cleaned_fix
                    )
                    active_code_target = cleaned_fix
                else:
                    candidate_full_code = cleaned_fix if cleaned_fix else request.code
            except Exception as gen_err:
                candidate_full_code = request.code

            # Syntax Check
            is_valid, _, _ = self.analyzer.inspect_syntax(candidate_full_code)
            is_syntax_valid = is_valid

            # Subprocess Sandbox Execution
            s_passed, s_out, s_err = self._run_sandbox(candidate_full_code)

            attempts_history.append(
                IterationAttempt(
                    attempt_number=attempt,
                    patch_code=candidate_full_code,
                    passed=s_passed,
                    error_summary=s_err.splitlines()[-1] if s_err.strip() else ("OK" if s_passed else "Unknown Error")
                )
            )

            final_suggested_fix = candidate_full_code
            final_passed = s_passed
            final_stdout = s_out
            final_stderr = s_err

            if s_passed:
                break
            else:
                err_lines = [l.strip() for l in s_err.split("\n") if l.strip()]
                if err_lines:
                    current_error = err_lines[-1]

        # Step 4: Confidence Scoring
        confidence = 0.50
        if is_syntax_valid:
            confidence += 0.20
        if final_passed:
            confidence += 0.25
        if len(attempts_history) > 1 and final_passed:
            confidence += 0.05

        err_type = raw_error.split(":")[0].strip() if ":" in raw_error else "RuntimeError"

        return DebugResponse(
            error_type=err_type,
            error_location=location_desc,
            root_cause=raw_error,
            suggested_fix=final_suggested_fix,
            syntax_valid=bool(is_syntax_valid),
            confidence_score=min(1.0, round(confidence, 2)),
            sandbox_executed=True,
            sandbox_passed=final_passed,
            sandbox_stdout=final_stdout,
            sandbox_stderr=final_stderr,
            iterations_used=len(attempts_history),
            self_corrected=bool(len(attempts_history) > 1 and final_passed),
            attempts=attempts_history
        )


debug_service = DebugService()