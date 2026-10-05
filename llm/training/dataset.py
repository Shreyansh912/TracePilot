"""
Loads raw code text, tokenizes it into a contiguous PyTorch 1D tensor,
and samples input (X) and target (Y) batches for autoregressive modeling.
"""
from typing import Tuple
import torch


class CodeDataset:
    def __init__(self, data_path: str, tokenizer, split_ratio: float = 0.9):
        """
        Loads text data, tokenizes into a 1D tensor, and splits into train/val.
        """
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize entire corpus into 1D LongTensor
        encoded_ids = tokenizer.encode(text)
        self.data = torch.tensor(encoded_ids, dtype=torch.long)

        # Split into train and validation sets
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
        """
        Generates a small batch of inputs X and targets Y.
        
        Args:
            split: 'train' or 'val'
            batch_size (B): number of independent sequences to process in parallel
            block_size (T): maximum context length (time dimension)
            device: torch device ('cuda' or 'cpu')
            
        Returns:
            X: tensor of shape (B, T)
            Y: tensor of shape (B, T)
        """
        data = self.train_data if split == "train" else self.val_data
        
        # Guard check: Ensure corpus is large enough for context length
        if len(data) <= block_size:
            raise ValueError(
                f"Data length ({len(data)}) must be strictly greater than block_size ({block_size})."
            )

        # Randomly choose start positions for each sequence in the batch
        # Maximum valid start index is len(data) - block_size - 1
        max_idx = len(data) - block_size
        idx = torch.randint(0, max_idx, (batch_size,))

        # Slice X and Y
        x = torch.stack([data[i : i + block_size] for i in idx])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])

        # Pin memory transfer to GPU
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        return x, y