"""
Fine-tunes the base Code Transformer on debugging and patch generation.
Features:
- Loads pretrained CodeGPT weights
- Evaluates masked cross-entropy loss exclusively on fixes/explanations
- Uses reduced learning rate for transfer learning (5e-5)
- Saves best debugging model checkpoint
"""
import sys
import os
import time
from typing import Dict
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer
from llm.model.transformer import MiniGPT
from llm.training.debug_dataset import DebuggingDataset
from llm.training.train import configure_optimizers


def finetune():
    base_checkpoint = os.path.join("llm", "checkpoints", "best_codegpt.pt")
    dataset_path = os.path.join("datasets", "processed", "debugging_dataset.jsonl")
    bpe_dir = os.path.join("llm", "tokenizer", "bpe_model")
    save_path = os.path.join("llm", "checkpoints", "best_debugger.pt")

    assert os.path.exists(base_checkpoint), f"Pretrained base model missing at {base_checkpoint}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Fine-Tuning on: {device}")

    # Load Tokenizer & Pretrained Model
    tokenizer = CodeBPETokenizer(tokenizer_dir=bpe_dir)
    checkpoint = torch.load(base_checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = MiniGPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=0.1
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[+] Loaded pretrained weights from {base_checkpoint} (Base Step: {checkpoint['step']})")

    # Prepare Dataset & DataLoader
    dataset = DebuggingDataset(dataset_path, tokenizer, max_seq_len=config["block_size"])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Transfer learning hyperparameters: lower LR (5e-5) to avoid catastrophic forgetting
    learning_rate = 5e-5
    optimizer = configure_optimizers(model, weight_decay=1e-2, learning_rate=learning_rate, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    epochs = 40
    print(f"[*] Commencing Fine-Tuning for {epochs} epochs over {len(dataset)} debugging pairs...")

    model.train()
    start_time = time.time()
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0

        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch + 1:2d}/{epochs}] | Masked Cross-Entropy Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_loss,
                    "config": config
                }, save_path)
                print(f"    -> Saved fine-tuned checkpoint (Loss: {best_loss:.4f})")

    elapsed = time.time() - start_time
    print(f"\n[+] Fine-tuning completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    finetune()