"""
scripts/test_backend_step9.py
Tests the FastAPI backend by sending a direct in-process HTTP request
simulating a user debugging session.
"""
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.main import app


def test_backend_debug_endpoint():
    client = TestClient(app)

    print("[*] 1. Testing /health endpoint...")
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print(f"[+] Health Response: {res.json()}")

    print("\n[*] 2. Testing /debug endpoint with ZeroDivisionError...")
    payload = {
        "code": "def divide_numbers(a, b):\n    return a / b",
        "language": "python",
        "error_message": "ZeroDivisionError: division by zero"
    }

    res = client.post("/debug", json=payload)
    assert res.status_code == 200, f"Debug endpoint failed: {res.text}"

    data = res.json()
    print("[+] Debug Endpoint Success!")
    print(f"    Error Type:       {data['error_type']}")
    print(f"    Location:         {data['error_location']}")
    print(f"    Root Cause:       {data['root_cause']}")
    print(f"    Syntax Valid:     {data['syntax_valid']}")
    print(f"    Confidence:       {data['confidence_score']}")
    print(f"    Suggested Fix:\n{data['suggested_fix']}")


if __name__ == "__main__":
    test_backend_debug_endpoint()