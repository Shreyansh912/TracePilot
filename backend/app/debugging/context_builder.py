"""
backend/app/debugging/context_builder.py
Uses Python AST to extract the exact function or class enclosing an offending line number,
pruning irrelevant file context so it fits cleanly inside small Transformer context windows.
"""
import ast
from typing import Optional, Dict, Any


class ContextBuilder:
    @staticmethod
    def extract_enclosing_scope(code: str, target_line: Optional[int] = None) -> str:
        """
        Extracts the function or code block containing target_line.
        If line is not provided or parsing fails, returns full code.
        """
        if not target_line:
            return code

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code  # Fallback to full code if syntax tree cannot be formed

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # ast nodes have lineno and end_lineno in Python 3.8+
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start and end and start <= target_line <= end:
                    lines = code.splitlines()
                    # Slices target function
                    return "\n".join(lines[start - 1 : end])

        return code

    @classmethod
    def assemble_prompt(cls, code: str, raw_trace: Optional[str] = None, error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Merges raw user code, stack trace, and AST extraction into a compact prompt payload.
        """
        from backend.app.debugging.traceback_parser import TracebackParser

        parsed_trace = TracebackParser.parse(raw_trace) if raw_trace else {}
        target_line = parsed_trace.get("line")
        error_type = parsed_trace.get("error_type") or "Error"
        err_msg = parsed_trace.get("error_message") or error_message or "Unknown failure"

        focused_code = cls.extract_enclosing_scope(code, target_line)

        # Standardized TracePilot instruction format
        compact_prompt = f"<|code|>\n{focused_code}\n<|error|>\n{error_type}: {err_msg}\n<|fix|>\n"

        return {
            "prompt": compact_prompt,
            "focused_code": focused_code,
            "target_line": target_line,
            "error_type": error_type,
            "error_message": err_msg
        }