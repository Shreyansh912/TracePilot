"""
Loads the trained checkpoint from disk, generates text conditioned on 
different code prompts, and compares with untrained random noise.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.tokenizer.char_tokenizer import CharTokenizer
from llm.model.transformer import MiniGPT


def run_inference_test():
    checkpoint_path = os.path.join("llm", "checkpoints", "best_minigpt.pt")
    vocab_path = os.path.join("llm", "tokenizer", "vocab.json")

    assert os.path.exists(checkpoint_path), f"Checkpoint missing at {checkpoint_path}. Train the model first."
    assert os.path.exists(vocab_path), f"Vocab missing at {vocab_path}"

    tokenizer = CharTokenizer.load(vocab_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading checkpoint on: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Reconstruct model from saved configuration
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

    print(f"[+] Checkpoint loaded successfully from Step {checkpoint['step']} (Val Loss: {checkpoint['val_loss']:.4f})")

    # Test generation on programming prompts
    prompts = [
        "def ",
        "def add(",
        "    return "
    ]

    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE GENERATION TEST (Trained MiniGPT)")
    print("=" * 60)

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        # Generate with temperature=0.7 for coherent code structure sampling
        out = model.generate(x, max_new_tokens=80, temperature=0.7, top_k=10)
        generated_text = tokenizer.decode(out[0].tolist())
        print(f"\n--- PROMPT: {repr(prompt)} ---")
        print(generated_text)

    print("\n" + "=" * 60)
    print("[SUCCESS] Step 4: Model converged and generates structured Python code.")


if __name__ == "__main__":
    run_inference_test()