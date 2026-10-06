"""
llm/evaluation/eval_debugger.py
Evaluates the fine-tuned debugging model and inspects raw token outputs.
"""
import sys
import os
import ast
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer
from llm.model.transformer import MiniGPT


def test_debugging_inference():
    checkpoint_path = os.path.join("llm", "checkpoints", "best_debugger.pt")
    bpe_dir = os.path.join("llm", "tokenizer", "bpe_model")

    assert os.path.exists(checkpoint_path), f"Checkpoint missing at {checkpoint_path}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Evaluation on: {device}")

    tokenizer = CodeBPETokenizer(tokenizer_dir=bpe_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = MiniGPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=0.0
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Held-out evaluation case: ZeroDivisionError
    test_code = "def divide_numbers(a, b):\n    return a / b"
    test_error = "ZeroDivisionError: division by zero"

    prompt = f"<|code|>\n{test_code}\n<|error|>\n{test_error}\n<|fix|>\n"
    prompt_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt_ids)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print("\n" + "=" * 60)
    print(f"INPUT PROMPT ({prompt_len} tokens):")
    print(prompt)
    print("=" * 60)

    # Generate new tokens
    out = model.generate(x, max_new_tokens=64, temperature=0.7, top_k=10)
    all_tokens = out[0].tolist()
    generated_tokens = all_tokens[prompt_len:]  # Only tokens generated beyond prompt

    print(f"[*] Raw Generated Token IDs: {generated_tokens}")

    # Decode only the newly generated tokens
    completion = tokenizer.decode(generated_tokens)

    print("\n[MODEL RAW OUTPUT]:")
    print(repr(completion))
    print("\n[FORMATTED COMPLETION]:")
    print(completion.replace("<|eos|>", "").strip())
    print("=" * 60)

    # Programmatic Syntax Evaluation via Python's AST
    print("\n[*] Running AST Syntax Verification...")
    clean_text = completion.replace("<|eos|>", "").strip()
    try:
        # Extract lines that look like Python definitions or indentations
        code_lines = [
            line for line in clean_text.split("\n")
            if any(line.strip().startswith(kw) for kw in ["def ", "return ", "if ", "raise ", "try:", "except "])
            or line.startswith("    ")
        ]
        candidate_code = "\n".join(code_lines)
        if candidate_code:
            ast.parse(candidate_code)
            print(f"[+] AST Syntax Check: PASSED\n--- Parsed Code ---\n{candidate_code}")
        else:
            print("[!] Note: No recognizable Python block detected in output.")
    except SyntaxError as e:
        print(f"[-] AST Syntax Check: FAILED ({e})")


if __name__ == "__main__":
    test_debugging_inference()