"""
llm/training/train.py
Main training loop for MiniGPT on the scaled AST debugging dataset.
"""
import sys
import os
import math
import time
import json
from typing import Tuple, Dict, List
import torch
import torch.nn as nn

# Ensure project root is on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer
from llm.model.transformer import MiniGPT


class ScaledDebugDataset:
    """
    Loads JSONL debug records, formats them with delimiters,
    and supplies batched tensor sequences for training and validation.
    """
    def __init__(self, jsonl_path: str, tokenizer: CodeBPETokenizer, split_ratio: float = 0.85):
        if not os.path.isabs(jsonl_path):
            jsonl_path = os.path.join(ROOT_DIR, jsonl_path)

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset not found at {jsonl_path}. Run scripts/mutate_dataset.py first.")

        all_tokens = []
        record_count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = (
                    f"<|code|>\n{record['buggy_code']}\n"
                    f"<|error|>\n{record.get('error_message', record.get('error_trace', 'Error'))}\n"
                    f"<|fix|>\n{record['fixed_code']}<|eos|>\n"
                )
                encoded = tokenizer.encode(prompt)
                if encoded:
                    all_tokens.extend(encoded)
                    record_count += 1

        print(f"[*] Loaded {record_count} records from {os.path.basename(jsonl_path)}")
        if not all_tokens:
            raise ValueError(
                f"No tokens produced from {jsonl_path}. "
                "Ensure scripts/mutate_dataset.py produced valid records and BPE vocab matches."
            )

        self.data = torch.tensor(all_tokens, dtype=torch.long)
        n = int(split_ratio * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split: str, batch_size: int, block_size: int, device: torch.device):
        data = self.train_data if split == "train" else self.val_data
        if len(data) <= block_size:
            raise ValueError(f"Dataset slice ({len(data)}) is too small for block_size ({block_size}).")
        
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)


def configure_optimizers(model: MiniGPT, weight_decay: float, learning_rate: float, betas: Tuple[float, float]) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
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
    dataset: ScaledDebugDataset,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device
) -> Dict[str, float]:
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
    # ------------------ Hyperparameters & Paths ------------------
    data_path = os.path.join(ROOT_DIR, "datasets", "synthetic_debugging_dataset_scaled.jsonl")
    bpe_dir = os.path.join(ROOT_DIR, "llm", "tokenizer")
    checkpoint_dir = os.path.join(ROOT_DIR, "llm", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 4.9M Parameter Model Specifications
    block_size = 256
    n_layer = 6
    n_head = 6
    n_embd = 384
    dropout = 0.1

    # Training Hyperparameters
    batch_size = 16
    max_steps = 1500
    eval_interval = 250
    eval_iters = 25
    warmup_steps = 150
    max_lr = 5e-4
    min_lr = 5e-5
    weight_decay = 0.01
    grad_clip = 1.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training MiniGPT on backend: {device}")

    # Initialize Tokenizer and Scaled Dataset
    tokenizer = CodeBPETokenizer(tokenizer_dir=bpe_dir)
    dataset = ScaledDebugDataset(data_path, tokenizer, split_ratio=0.85)
    print(f"[+] Total tokens in corpus: {len(dataset.data):,} (Train: {len(dataset.train_data):,}, Val: {len(dataset.val_data):,})")

    # Initialize Transformer Architecture
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout
    ).to(device)

    optimizer = configure_optimizers(model, weight_decay, max_lr, (0.9, 0.95))

    print(f"[+] Model parameters: {model.get_num_params():,}")
    print("[*] Starting training loop...\n")

    best_val_loss = float("inf")
    start_time = time.time()

    for step in range(max_steps):
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % eval_interval == 0 or step == max_steps - 1:
            metrics = estimate_loss(model, dataset, eval_iters, batch_size, block_size, device)
            print(
                f"Step {step:4d} | "
                f"Train Loss: {metrics['train']:.4f} (PPL: {metrics['train_ppl']:.2f}) | "
                f"Val Loss: {metrics['val']:.4f} (PPL: {metrics['val_ppl']:.2f}) | "
                f"LR: {lr:.2e}"
            )

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

        # Fetch batch and execute forward/backward pass
        xb, yb = dataset.get_batch("train", batch_size, block_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    total_time = time.time() - start_time
    print(f"\n[+] Training complete in {total_time:.2f} seconds ({total_time / max_steps * 1000:.1f} ms/step).")


if __name__ == "__main__":
    train()