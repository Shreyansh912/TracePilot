"""
scripts/test_model_step3.py
Tests the MiniGPT architecture:
1. Instantiation and parameter counting.
2. Tensor shape flow through attention, blocks, and output head.
3. Loss calculation smoke test.
4. Autoregressive token generation smoke test.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.tokenizer.char_tokenizer import CharTokenizer
from llm.model.transformer import MiniGPT


def run_step3_test():
    vocab_path = os.path.join("llm", "tokenizer", "vocab.json")
    assert os.path.exists(vocab_path), "Run step 2 first to create vocab.json"

    tokenizer = CharTokenizer.load(vocab_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Testing MiniGPT on device: {device}")

    # Architectural Hyperparameters for our Mini Transformer
    vocab_size = tokenizer.vocab_size
    block_size = 64
    n_layer = 4
    n_head = 4
    n_embd = 128

    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0
    ).to(device)

    total_params = model.get_num_params()
    print(f"[+] Model instantiated successfully.")
    print(f"[+] Total Trainable Parameters: {total_params:,} (~{total_params / 1e6:.2f}M)")

    # 1. Forward Pass Shape Verification
    B, T = 2, 16
    dummy_input = torch.randint(0, vocab_size, (B, T), device=device)
    dummy_targets = torch.randint(0, vocab_size, (B, T), device=device)

    logits, loss = model(dummy_input, dummy_targets)

    print(f"\n[*] Forward Pass Tensor Shapes:")
    print(f"    Input Shape:   {dummy_input.shape}")
    print(f"    Logits Shape:  {logits.shape} (Expected: [{B}, {T}, {vocab_size}])")
    print(f"    Loss Value:    {loss.item():.4f}")

    assert logits.shape == (B, T, vocab_size), f"Shape mismatch: {logits.shape}"
    assert loss is not None and not torch.isnan(loss), "Loss computation failed or returned NaN!"
    print("[+] Forward pass and loss calculation verified.")

    # 2. Backward Pass Gradient Check
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"[+] Backward pass verified. Cumulative gradient norm: {grad_norm:.4f}")
    assert grad_norm > 0.0, "Gradients failed to propagate to parameters!"

    # 3. Text Generation Smoke Test (from initial prompt)
    prompt = "def "
    prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    print(f"\n[*] Generation test with prompt: {repr(prompt)}")

    generated_indices = model.generate(prompt_tokens, max_new_tokens=30, temperature=1.0)
    generated_text = tokenizer.decode(generated_indices[0].tolist())
    print(f"    Generated output (Untrained weights):")
    print(f"    {repr(generated_text)}")
    print("\n[SUCCESS] Step 3: Transformer architecture, forward/backward, and generation verified.")


if __name__ == "__main__":
    run_step3_test()