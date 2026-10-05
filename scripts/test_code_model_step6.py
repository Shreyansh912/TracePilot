"""
Verifies the trained Code Transformer:
1. Loads the checkpoint from disk.
2. Checks VRAM memory consumption.
3. Tests autoregressive completion on programming prompts using BPE subwords.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer
from llm.model.transformer import MiniGPT


def test_code_model():
    checkpoint_path = os.path.join("llm", "checkpoints", "best_codegpt.pt")
    bpe_dir = os.path.join("llm", "tokenizer", "bpe_model")

    assert os.path.exists(checkpoint_path), f"Checkpoint missing at {checkpoint_path}. Train the model first."
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating Code Transformer on: {device}")

    tokenizer = CodeBPETokenizer(tokenizer_dir=bpe_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device)
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

    print(f"[+] Loaded CodeGPT (Step {checkpoint['step']}, Val Loss: {checkpoint['val_loss']:.4f})")
    
    if device.type == "cuda":
        mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        print(f"[+] Active VRAM for Model: {mem_allocated:.2f} MB")

    test_prompts = [
        "def binary_search(",
        "class Node:\n    def __init__(",
        "def safe_divide("
    ]

    print("\n" + "=" * 60)
    print("CODE TRANSFORMER GENERATION SAMPLING")
    print("=" * 60)

    for prompt in test_prompts:
        token_ids = tokenizer.encode(prompt)
        x = torch.tensor([token_ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=60, temperature=0.6, top_k=8)
        decoded = tokenizer.decode(out[0].tolist())
        print(f"\n[PROMPT]:\n{prompt}")
        print(f"[COMPLETION]:\n{decoded}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Step 6: Programming-specialized model verified.")


if __name__ == "__main__":
    test_code_model()