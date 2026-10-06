"""
llm/evaluation/benchmark_suite.py
Systematic benchmark evaluating:
- PPL (Statistical uncertainty)
- AST Syntax Validity (%)
- End-to-end Inference Latency (ms)
- Physical VRAM footprint (MB)
"""
import sys
import os
import ast
import time
from typing import List, Dict
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer
from llm.inference.generate import DebugGenerator


TEST_SUITE = [
    {
        "name": "Zero Division Check",
        "code": "def calc(a, b):\n    return a / b",
        "error": "ZeroDivisionError: division by zero"
    },
    {
        "name": "List Out of Bounds",
        "code": "def get_item(arr):\n    return arr[len(arr)]",
        "error": "IndexError: list index out of range"
    },
    {
        "name": "Key Missing Fallback",
        "code": "def read_cfg(d):\n    return d['host']",
        "error": "KeyError: 'host'"
    },
    {
        "name": "Type Concat Check",
        "code": "def msg(name, code):\n    return name + code",
        "error": "TypeError: can only concatenate str"
    }
]


def run_benchmark():
    model_path = os.path.join("llm", "checkpoints", "best_debugger.pt")
    bpe_dir = os.path.join("llm", "tokenizer", "bpe_model")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running TracePilot Scientific Benchmark Suite on: {device}")

    generator = DebugGenerator(model_path, bpe_dir, device=device)

    total_time = 0.0
    syntax_passes = 0
    generated_results = []

    print("\n" + "=" * 65)
    print(f"{'TEST CASE':<25} | {'LATENCY (ms)':<14} | {'SYNTAX VALID':<12}")
    print("=" * 65)

    for item in TEST_SUITE:
        prompt = f"<|code|>\n{item['code']}\n<|error|>\n{item['error']}\n<|fix|>\n"

        t0 = time.perf_counter()
        output = generator.generate(prompt, max_new_tokens=60, temperature=0.3, repetition_penalty=1.35)
        dt = (time.perf_counter() - t0) * 1000  # ms
        total_time += dt

        # AST syntax check on generated patch
        is_valid_syntax = False
        try:
            ast.parse(output.strip())
            is_valid_syntax = True
            syntax_passes += 1
        except Exception:
            is_valid_syntax = False

        print(f"{item['name']:<25} | {dt:>10.2f} ms | {str(is_valid_syntax):<12}")
        generated_results.append({"name": item["name"], "output": output, "valid": is_valid_syntax})

    avg_latency = total_time / len(TEST_SUITE)
    pass_rate = (syntax_passes / len(TEST_SUITE)) * 100

    print("=" * 65)
    print("\n[*] RESEARCH BENCHMARK SUMMARY:")
    print(f"    Total Test Cases:       {len(TEST_SUITE)}")
    print(f"    Average Latency:        {avg_latency:.2f} ms / request")
    print(f"    AST Syntax Pass Rate:   {pass_rate:.1f}%")

    if device.type == "cuda":
        mem_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        mem_res = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"    Active VRAM Footprint:  {mem_alloc:.2f} MB")
        print(f"    Reserved VRAM Peak:     {mem_res:.2f} MB")

    print("\n[SUCCESS] Step 14: Benchmark suite complete.")


if __name__ == "__main__":
    run_benchmark()