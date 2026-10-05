"""
Tokenizes code corpora using the trained BPE model.
Supports slicing subword token streams into (B, T) batches.
"""
from typing import Tuple
import torch
from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer


class CodeBPEDataset:
    def __init__(self, data_path: str, tokenizer: CodeBPETokenizer, split_ratio: float = 0.85):
        self.tokenizer = tokenizer
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize entire corpus using BPE
        encoded = tokenizer.encode(text)
        self.data = torch.tensor(encoded, dtype=torch.long)

        # Train / Validation Split
        n_train = int(len(self.data) * split_ratio)
        self.train_data = self.data[:n_train]
        self.val_data = self.data[n_train:]

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data

        if len(data) <= block_size:
            raise ValueError(
                f"Data split length ({len(data)}) must be larger than block_size ({block_size})."
            )

        max_idx = len(data) - block_size
        idx = torch.randint(0, max_idx, (batch_size,))

        x = torch.stack([data[i : i + block_size] for i in idx])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        return x, y