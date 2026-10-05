"""
backend/app/sandbox/runner.py
Isolated code execution runner with static security audits,
execution timeouts, and stdout/stderr capture.
"""
import ast
import os
import sys
import tempfile
import subprocess
from typing import Dict, Any, List

DANGEROUS_MODULES = {"os", "subprocess", "socket", "ctypes", "sys", "shutil"}
DANGEROUS_BUILTINS = {"eval", "exec", "__import__", "compile"}


class SandboxSecurityError(Exception):
    pass


class CodeSandbox:
    def __init__(self, timeout_seconds: float = 3.0):
        self.timeout_seconds = timeout_seconds

    def audit_ast(self, code: str) -> None:
        """
        Statically inspects the AST to block dangerous system-level functions.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SandboxSecurityError(f"Syntax error during security audit: {e.msg}")

        for node in ast.walk(tree):
            # Check module imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    if root_pkg in DANGEROUS_MODULES:
                        raise SandboxSecurityError(f"Security Alert: Import of forbidden module '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split(".")[0]
                    if root_pkg in DANGEROUS_MODULES:
                        raise SandboxSecurityError(f"Security Alert: Import from forbidden module '{node.module}'")
            # Check direct calls to dangerous builtins
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_BUILTINS:
                    raise SandboxSecurityError(f"Security Alert: Call to forbidden function '{node.func.id}()'")

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Runs user code inside a temporary directory with timeout protection.
        """
        # Step 1: Security Audit
        try:
            self.audit_ast(code)
        except SandboxSecurityError as sec_err:
            return {
                "success": False,
                "error_type": "SecurityViolation",
                "stderr": str(sec_err),
                "stdout": "",
                "timed_out": False
            }

        # Step 2: Temporary File Allocation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_target.py")
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)

            # Scrub environment variables to prevent leaking server secrets
            safe_env = {
                "PYTHONPATH": "",
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "PATH": os.environ.get("PATH", "")
            }

            # Step 3: Run in isolated subprocess
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", temp_file],  # -I flag isolates from user site-packages
                    cwd=temp_dir,
                    env=safe_env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )
                return {
                    "success": proc.returncode == 0,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "timed_out": False
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Execution timed out after {self.timeout_seconds} seconds.",
                    "timed_out": True
                }
            except Exception as e:
                return {
                    "success": False,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Subprocess runner error: {str(e)}",
                    "timed_out": False
                }