"""
backend/app/debugging/analyzer.py
Deterministic static code analyzer for Python.
Parses Abstract Syntax Trees (AST) to detect syntax anomalies,
missing colons, or invalid structures prior to model invocation.
"""
import ast
from typing import Optional, Tuple


class CodeAnalyzer:
    @staticmethod
    def inspect_syntax(source_code: str) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Parses source code into an AST.
        Returns:
            (is_valid, error_description, line_number)
        """
        try:
            ast.parse(source_code)
            return True, None, None
        except SyntaxError as e:
            return False, f"{e.msg} ({type(e).__name__})", e.lineno
        except Exception as e:
            return False, str(e), None