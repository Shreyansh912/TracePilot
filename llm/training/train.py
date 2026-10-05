"""
Main training loop for the MiniGPT model:
- Parameter-grouped AdamW optimizer (weight decay only on 2D weights)
- Linear warmup + Cosine learning rate decay scheduler
- Gradient clipping
- Periodic validation loss and perplexity evaluation
- Checkpoint persistence
"""
import sys
import os
import math
import time
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llm.tokenizer.char_tokenizer import CharTokenizer
from llm.model.transformer import MiniGPT
from llm.training.dataset import CodeDataset


def configure_optimizers(model: MiniGPT, weight_decay: float, learning_rate: float, betas: Tuple[float, float]) -> torch.optim.Optimizer:
    """
    Separates parameters into those that decay (weights of Linear layers/embeddings)
    and those that do not (biases and LayerNorm scales).
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and 1D parameters (LayerNorm weights) do not undergo weight decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Computes learning rate via linear warmup followed by cosine annealing decay.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    dataset: CodeDataset,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device
) -> Dict[str, float]:
    """
    Averages cross-entropy loss and perplexity across multiple validation iterations.
    """
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()
        out[split] = mean_loss
        out[f"{split}_ppl"] = math.exp(mean_loss)
    model.train()
    return out


def train():
    # ------------------ Hyperparameters ------------------
    data_path = os.path.join("datasets", "raw", "tiny_code.txt")
    vocab_path = os.path.join("llm", "tokenizer", "vocab.json")
    checkpoint_dir = os.path.join("llm", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Architectural settings
    block_size = 64
    n_layer = 4
    n_head = 4
    n_embd = 128
    dropout = 0.1

    # Training parameters
    batch_size = 16
    max_steps = 1200
    eval_interval = 200
    eval_iters = 20
    warmup_steps = 100
    max_lr = 6e-4
    min_lr = 6e-5
    weight_decay = 1e-2
    grad_clip = 1.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training MiniGPT on compute backend: {device}")

    # Load tokenizer and dataset
    tokenizer = CharTokenizer.load(vocab_path)
    dataset = CodeDataset(data_path, tokenizer, split_ratio=0.85)
    print(f"[+] Corpus size: {len(dataset.data)} tokens (Train: {len(dataset.train_data)}, Val: {len(dataset.val_data)})")

    # Initialize model
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout
    ).to(device)

    optimizer = configure_optimizers(model, weight_decay, max_lr, (0.9, 0.95))

    print(f"[+] Total trainable parameters: {model.get_num_params():,}")
    print("[*] Commencing training loop...\n")

    best_val_loss = float("inf")
    start_time = time.time()

    for step in range(max_steps):
        # Update learning rate per schedule
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Periodic Evaluation & Validation
        if step % eval_interval == 0 or step == max_steps - 1:
            metrics = estimate_loss(model, dataset, eval_iters, batch_size, block_size, device)
            print(
                f"Step {step:4d} | "
                f"Train Loss: {metrics['train']:.4f} (PPL: {metrics['train_ppl']:.2f}) | "
                f"Val Loss: {metrics['val']:.4f} (PPL: {metrics['val_ppl']:.2f}) | "
                f"LR: {lr:.2e}"
            )

            # Save best checkpoint
            if metrics["val"] < best_val_loss:
                best_val_loss = metrics["val"]
                checkpoint_path = os.path.join(checkpoint_dir, "best_minigpt.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "block_size": block_size,
                        "n_layer": n_layer,
                        "n_head": n_head,
                        "n_embd": n_embd
                    }
                }, checkpoint_path)
                print(f"    -> Checkpoint saved with new best Val Loss: {best_val_loss:.4f}")

        # Fetch training batch
        xb, yb = dataset.get_batch("train", batch_size, block_size, device)

        # Forward pass
        _, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        optimizer.step()

    total_time = time.time() - start_time
    print(f"\n[+] Training complete in {total_time:.2f} seconds ({total_time / max_steps * 1000:.1f} ms/step).")


if __name__ == "__main__":
    train()