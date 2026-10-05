"""
Trains the ~5M parameter programming-specialized Transformer on BPE tokens.
Includes:
- Mixed precision training with torch.cuda.amp (autocast + GradScaler)
- AdamW with decoupled weight decay
- Learning rate warmup + Cosine Annealing
- Periodic validation loss and perplexity tracking
- Checkpoint persistence
"""
import sys
import os
import math
import time
from typing import Dict
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer
from llm.model.transformer import MiniGPT
from llm.training.bpe_dataset import CodeBPEDataset
from llm.training.train import configure_optimizers, get_lr


@torch.no_grad()
def evaluate_metrics(
    model: MiniGPT,
    dataset: CodeBPEDataset,
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
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                _, loss = model(x, y)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()
        out[split] = mean_loss
        out[f"{split}_ppl"] = math.exp(mean_loss)
    model.train()
    return out


def train():
    # ------------------ Hyperparameters ------------------
    data_path = os.path.join("datasets", "raw", "python_code_corpus.txt")
    bpe_dir = os.path.join("llm", "tokenizer", "bpe_model")
    checkpoint_dir = os.path.join("llm", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Architectural specs (~4.9M params)
    block_size = 128
    n_layer = 6
    n_head = 8
    n_embd = 256
    dropout = 0.1

    # Training specs
    batch_size = 8
    max_steps = 1000
    eval_interval = 200
    eval_iters = 15
    warmup_steps = 100
    max_lr = 5e-4
    min_lr = 5e-5
    weight_decay = 1e-2
    grad_clip = 1.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Code Transformer on compute backend: {device}")

    # Load BPE Tokenizer and Dataset
    tokenizer = CodeBPETokenizer(tokenizer_dir=bpe_dir)
    dataset = CodeBPEDataset(data_path, tokenizer, split_ratio=0.85)

    print(f"[+] Total BPE tokens in corpus: {len(dataset.data)}")
    print(f"    Train split: {len(dataset.train_data)} tokens | Val split: {len(dataset.val_data)} tokens")

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout
    ).to(device)

    total_params = model.get_num_params()
    print(f"[+] Trainable Parameters: {total_params:,} (~{total_params / 1e6:.2f}M)")

    optimizer = configure_optimizers(model, weight_decay, max_lr, (0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    start_time = time.time()
    print("[*] Commencing training loop with Mixed Precision (AMP)...\n")

    for step in range(max_steps):
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Periodic Validation
        if step % eval_interval == 0 or step == max_steps - 1:
            metrics = evaluate_metrics(model, dataset, eval_iters, batch_size, block_size, device)
            print(
                f"Step {step:4d} | "
                f"Train Loss: {metrics['train']:.4f} (PPL: {metrics['train_ppl']:.2f}) | "
                f"Val Loss: {metrics['val']:.4f} (PPL: {metrics['val_ppl']:.2f}) | "
                f"LR: {lr:.2e}"
            )

            if metrics["val"] < best_val_loss:
                best_val_loss = metrics["val"]
                checkpoint_path = os.path.join(checkpoint_dir, "best_codegpt.pt")
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
                print(f"    -> Best model checkpoint saved to: {checkpoint_path}")

        # Fetch Batch
        xb, yb = dataset.get_batch("train", batch_size, block_size, device)

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            _, loss = model(xb, yb)

        # Scaled Backward Pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Scaler Optimizer Step
        scaler.step(optimizer)
        scaler.update()

    total_time = time.time() - start_time
    print(f"\n[+] Code Model Training complete in {total_time:.2f} seconds ({total_time / max_steps * 1000:.1f} ms/step).")


if __name__ == "__main__":
    train()