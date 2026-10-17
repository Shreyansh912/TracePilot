"""
backend/app/services/debug_service.py
Coordinates AST scope isolation, traceback symbol matching,
MiniGPT inference, AST semantic repair fallbacks, and sandbox execution.
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

    def _apply_ast_rule_repair(self, func_code: str, error_type: str) -> str:
        """
        Synthesizes a clean AST patch for runtime exception classes
        if the small language model generates syntactically invalid code.
        """
        try:
            tree = ast.parse(func_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # 1. ZeroDivisionError Guard
                    if "ZeroDivision" in error_type:
                        for stmt in node.body:
                            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.BinOp):
                                if isinstance(stmt.value.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                                    divisor = ast.unparse(stmt.value.right)
                                    guard = ast.parse(f"if {divisor} == 0:\n    return 0").body[0]
                                    node.body.insert(0, guard)
                                    return ast.unparse(tree)

                    # 2. IndexError Guard
                    elif "IndexError" in error_type:
                        for stmt in node.body:
                            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Subscript):
                                target = ast.unparse(stmt.value.value)
                                guard = ast.parse(f"if not {target}:\n    return None").body[0]
                                node.body.insert(0, guard)
                                # Cap index access
                                stmt.value.slice = ast.parse(f"min(len({target}) - 1, max(0, 0))").body[0].value
                                return ast.unparse(tree)

                    # 3. KeyError Guard
                    elif "KeyError" in error_type:
                        for stmt in node.body:
                            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Subscript):
                                d_target = ast.unparse(stmt.value.value)
                                k_slice = ast.unparse(stmt.value.slice)
                                return ast.unparse(ast.parse(func_code).body[0]).replace(
                                    f"{d_target}[{k_slice}]", f"{d_target}.get({k_slice}, None)"
                                )
        except Exception:
            pass
        return func_code

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

        raw_error = request.error_message or ""
        if not raw_error and request.stack_trace:
            lines = [l.strip() for l in request.stack_trace.strip().split("\n") if l.strip()]
            raw_error = lines[-1] if lines else "RuntimeError"
        current_error = raw_error.split("\n")[-1].strip()

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

        for attempt in range(1, max_attempts + 1):
            prompt = f"<|code|>\n{active_code_target}\n<|error|>\n{current_error}\n<|fix|>\n"
            print(f"\n[*] --- Attempt {attempt}/{max_attempts} Inference ---")

            candidate_patch = ""
            try:
                raw_generation = self.generator.generate(prompt=prompt, max_new_tokens=90)
                cleaned_fix = raw_generation.replace("<|eos|>", "").strip()
                
                # Check if neural output is valid Python code
                is_valid, _, _ = self.analyzer.inspect_syntax(cleaned_fix)
                if is_valid and "def " in cleaned_fix:
                    candidate_patch = cleaned_fix
                else:
                    print(f"[!] MiniGPT output has invalid syntax. Engaging AST Semantic Repair Engine.")
                    candidate_patch = self._apply_ast_rule_repair(active_code_target, current_error)
            except Exception as e:
                print(f"[!] Generation error: {e}. Falling back to AST Engine.")
                candidate_patch = self._apply_ast_rule_repair(active_code_target, current_error)

            if non_target_nodes:
                candidate_full_code = f"{candidate_patch}\n\n" + "\n".join(non_target_nodes)
            else:
                candidate_full_code = candidate_patch

            # Inspect AST
            is_valid, _, _ = self.analyzer.inspect_syntax(candidate_full_code)
            is_syntax_valid = is_valid

            # Execute in Subprocess Sandbox
            s_passed, s_out, s_err = self._run_sandbox(candidate_full_code)

            attempts_history.append(
                IterationAttempt(
                    attempt_number=attempt,
                    patch_code=candidate_full_code,
                    passed=s_passed,
                    error_summary=s_err.splitlines()[-1] if s_err.strip() else ("OK" if s_passed else "Runtime Error")
                )
            )

            final_suggested_fix = candidate_full_code
            final_passed = s_passed
            final_stdout = s_out
            final_stderr = s_err

            if s_passed:
                print(f"[+] Attempt {attempt} passed sandbox verification (Exit 0).")
                break
            else:
                err_lines = [l.strip() for l in s_err.split("\n") if l.strip()]
                if err_lines:
                    current_error = err_lines[-1]

        confidence = 0.50
        if is_syntax_valid:
            confidence += 0.20
        if final_passed:
            confidence += 0.25

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