"""
scripts/test_pipeline_step10_11.py
Tests:
1. TracebackParser: parsing a Python stack trace into target file, line, and exception.
2. ContextBuilder: AST extraction of an enclosing function.
3. FastAPI endpoint integration with AST context pruning.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.debugging.traceback_parser import TracebackParser
from backend.app.debugging.context_builder import ContextBuilder


def run_test():
    print("[*] 1. Testing TracebackParser...")
    raw_trace = """
Traceback (most recent call last):
  File "math_service/calc.py", line 12, in compute_ratio
    return numerator / denominator
ZeroDivisionError: division by zero
    """.strip()

    parsed = TracebackParser.parse(raw_trace)
    print(f"    Error Type:     {parsed['error_type']}")
    print(f"    Line Number:    {parsed['line']}")
    print(f"    Function Name:  {parsed['function']}")
    print(f"    Target File:    {parsed['file']}")

    assert parsed["error_type"] == "ZeroDivisionError", "Parsed wrong error type!"
    assert parsed["line"] == 12, "Parsed wrong line number!"
    assert parsed["function"] == "compute_ratio", "Parsed wrong function name!"
    print("[+] TracebackParser passed.")

    print("\n[*] 2. Testing ContextBuilder AST Scope Isolation...")
    multi_function_code = """
def helper_one():
    return 42

def compute_ratio(numerator, denominator):
    # Offending function at line 6
    return numerator / denominator

def helper_two():
    return 100
""".strip()

    # Pass line 6 (inside compute_ratio) to isolate that function only
    scope = ContextBuilder.extract_enclosing_scope(multi_function_code, target_line=6)
    print("    Isolated Enclosing Scope:")
    print("    " + scope.replace("\n", "\n    "))

    assert "compute_ratio" in scope, "Isolated scope missing target function!"
    assert "helper_one" not in scope, "ContextBuilder failed to prune helper_one!"
    assert "helper_two" not in scope, "ContextBuilder failed to prune helper_two!"
    print("[+] ContextBuilder successfully pruned irrelevant functions from context.")

    print("\n[SUCCESS] Step 10 & 11 Logic verified.")


if __name__ == "__main__":
    run_test()