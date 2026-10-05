"""
scripts/test_generator_step8.py
Verifies that repetition penalty eliminates degenerate token loops.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.inference.generate import DebugGenerator


def run_test():
    model_path = os.path.join("llm", "checkpoints", "best_debugger.pt")
    bpe_dir = os.path.join("llm", "tokenizer", "bpe_model")

    generator = DebugGenerator(model_path, bpe_dir)

    test_code = "def divide_numbers(a, b):\n    return a / b"
    test_error = "ZeroDivisionError: division by zero"
    prompt = f"<|code|>\n{test_code}\n<|error|>\n{test_error}\n<|fix|>\n"

    print("\n" + "=" * 60)
    print("PROMPT:")
    print(prompt.strip())
    print("=" * 60)

    completion = generator.generate(
        prompt=prompt,
        max_new_tokens=80,
        temperature=0.5,
        top_k=8,
        repetition_penalty=1.4
    )

    print("\n[GENERATED COMPLETION]:")
    print(completion.strip())
    print("=" * 60)


if __name__ == "__main__":
    run_test()