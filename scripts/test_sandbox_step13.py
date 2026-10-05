"""
scripts/test_sandbox_step13.py
Tests:
1. Safe subprocess execution of valid Python code.
2. Interception of infinite while loops via timeout.
3. Static AST blocking of forbidden imports (e.g. 'import os').
4. Automatic failure capture and structured output.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.sandbox.runner import CodeSandbox


def run_tests():
    sandbox = CodeSandbox(timeout_seconds=2.0)

    # 1. Test Valid Execution
    print("[*] 1. Testing clean code execution...")
    clean_code = "a = 10\nb = 20\nprint(f'RESULT={a + b}')"
    res1 = sandbox.execute(clean_code)
    print(f"    Success: {res1['success']}")
    print(f"    Stdout:  {repr(res1['stdout'].strip())}")
    assert res1["success"] is True
    assert "RESULT=30" in res1["stdout"]
    print("[+] Clean execution passed.")

    # 2. Test Infinite Loop Timeout
    print("\n[*] 2. Testing infinite loop timeout handling (2.0s limit)...")
    infinite_loop = "while True:\n    pass"
    res2 = sandbox.execute(infinite_loop)
    print(f"    Success:   {res2['success']}")
    print(f"    Timed Out: {res2['timed_out']}")
    print(f"    Stderr:    {repr(res2['stderr'])}")
    assert res2["timed_out"] is True
    print("[+] Timeout guard passed.")

    # 3. Test Security AST Audit: Block dangerous imports
    print("\n[*] 3. Testing security audit (Blocking 'import os')...")
    malicious_code = "import os\nos.listdir('.')"
    res3 = sandbox.execute(malicious_code)
    print(f"    Success:    {res3['success']}")
    print(f"    Error Type: {res3.get('error_type')}")
    print(f"    Stderr:     {repr(res3['stderr'])}")
    assert res3["error_type"] == "SecurityViolation"
    print("[+] Security audit successfully blocked unauthorized system import.")

    print("\n[SUCCESS] Step 13: Sandbox execution, security auditing, and timeouts operational.")


if __name__ == "__main__":
    run_tests()