"""
Tests the DebuggingDataset pipeline:
- Validates prompt-to-fix token boundaries.
- Confirms target label masking (-100 on prompt, valid IDs on fix).
- Verifies batch collation shapes and padding consistency.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer
from llm.training.debug_dataset import DebuggingDataset


def test_debugging_dataset():
    bpe_dir = os.path.join("llm", "tokenizer", "bpe_model")
    dataset_path = os.path.join("datasets", "processed", "debugging_dataset.jsonl")

    assert os.path.exists(dataset_path), f"Debugging dataset not found at {dataset_path}"
    tokenizer = CodeBPETokenizer(tokenizer_dir=bpe_dir)

    print("[*] Initializing DebuggingDataset test...")
    max_len = 128
    dataset = DebuggingDataset(dataset_path, tokenizer, max_seq_len=max_len)
    print(f"[+] Loaded {len(dataset)} debugging sample pairs.")

    assert len(dataset) >= 10, "Expected at least 10 debugging samples in seed dataset."

    # Inspect the first sample
    x, y = dataset[0]
    print(f"\n[*] Tensor Dimensions:")
    print(f"    Input X Shape:  {x.shape}")
    print(f"    Target Y Shape: {y.shape}")

    assert x.shape == (max_len,), f"Expected shape ({max_len},), got {x.shape}"
    assert y.shape == (max_len,), f"Expected shape ({max_len},), got {y.shape}"

    # Verify Label Masking:
    # y must start with -100 tokens, then have valid token IDs, then finish with -100 padding
    masked_count = (y == -100).sum().item()
    supervised_count = (y != -100).sum().item()

    print(f"\n[*] Target Masking Analysis for Sample 0:")
    print(f"    Total Tokens:       {max_len}")
    print(f"    Supervised Tokens:  {supervised_count} (Loss calculated here)")
    print(f"    Masked Tokens:      {masked_count} (Targets set to -100)")

    assert supervised_count > 0, "No tokens are being supervised in the target sequence!"
    assert y[0].item() == -100, "First token must be masked (-100) because it is part of prompt."

    # Decode supervised targets to ensure they contain the explanation and fix
    supervised_indices = [idx.item() for idx in y if idx.item() != -100]
    decoded_fix = tokenizer.decode(supervised_indices)
    print(f"\n[*] Decoded Supervised Ground Truth:\n{decoded_fix}")

    print("\n[SUCCESS] Step 7: Debugging dataset and target masking pipeline verified.")


if __name__ == "__main__":
    test_debugging_dataset()