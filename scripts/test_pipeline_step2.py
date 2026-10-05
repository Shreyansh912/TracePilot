"""
Tests character tokenization, round-trip encoding/decoding,
and verifies tensor shapes and target shifting.
"""
import sys
import os
import torch

# Ensure TracePilot root is in Python module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.tokenizer.char_tokenizer import CharTokenizer
from llm.training.dataset import CodeDataset


def run_pipeline_test():
    raw_data_path = os.path.join("datasets", "raw", "tiny_code.txt")
    vocab_save_path = os.path.join("llm", "tokenizer", "vocab.json")

    print("[*] Step 2 Test: Reading source corpus...")
    with open(raw_data_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    print(f"[+] Total characters loaded: {len(corpus)}")

    # 1. Initialize and fit tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(corpus)
    print(f"[+] Vocabulary fitted. Unique tokens (Vocab Size V): {tokenizer.vocab_size}")
    print(f"[+] Vocabulary Characters: {repr(''.join(tokenizer.chars))}")

    # 2. Test round-trip encode/decode
    test_str = "def foo(x):\n    return x + 1"
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    assert test_str == decoded, "Decode output did not match original input string!"
    print("[+] Round-trip encode -> decode verification passed.")

    # 3. Save tokenizer vocabulary
    tokenizer.save(vocab_save_path)
    print(f"[+] Vocabulary saved to {vocab_save_path}")

    # 4. Verify batch sampling and target alignment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = CodeDataset(raw_data_path, tokenizer, split_ratio=0.8)

    B = 4   # Batch size
    T = 16  # Block size (context length)
    X, Y = dataset.get_batch(split="train", batch_size=B, block_size=T, device=device)

    print(f"\n[*] Batch Tensor Shapes on device ({device}):")
    print(f"    X shape: {X.shape} (Expected: [{B}, {T}])")
    print(f"    Y shape: {Y.shape} (Expected: [{B}, {T}])")
    assert X.shape == (B, T), f"Incorrect X shape: {X.shape}"
    assert Y.shape == (B, T), f"Incorrect Y shape: {Y.shape}"

    # 5. Verify the target shift property: Y[b, t] must equal the token following X[b, t]
    print("\n[*] Verifying Autoregressive Target Shift:")
    sample_b = 0
    sample_x = X[sample_b].tolist()
    sample_y = Y[sample_b].tolist()

    print(f"    Input  X[0]: {repr(tokenizer.decode(sample_x))}")
    print(f"    Target Y[0]: {repr(tokenizer.decode(sample_y))}")

    # Check shift alignment: X[1:] must match Y[:-1]
    assert sample_x[1:] == sample_y[:-1], "Target tensor Y is not shifted by exactly +1 token!"
    print("[+] Shift verification passed: Targets match inputs shifted by exactly +1.")

    print("\n[SUCCESS] Step 2: Character tokenizer & dataset tensor pipeline operational.")


if __name__ == "__main__":
    run_pipeline_test()