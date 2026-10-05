"""
Decoder-only Transformer implemented from first principles in PyTorch.
Includes:
- Causal Self-Attention with Scaled Dot-Product
- Multi-Head Attention
- Pre-LN Transformer Blocks (GELU MLP + Residuals)
- Autoregressive Next-Token Loss computation
- Autoregressive Generation
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        # Key, Query, Value projections packaged into a single linear layer
        # Output shape: (B, T, 3 * n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower triangular matrix of ones
        # Shape: (1, 1, block_size, block_size)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Step 1: Linear projection into Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)  # each is (B, T, C)

        # Step 2: Reshape for multi-head attention: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Step 3: Scaled dot-product attention scores
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Step 4: Apply causal mask (fill future positions with -infinity)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # Step 5: Softmax normalization and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Step 6: Weighted sum of values
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = att @ v

        # Step 7: Re-assemble heads: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Step 8: Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        # Expansion factor of 4 is standard in modern Transformer architectures
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, T, 4*C) -> (B, T, C)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        # Pre-LN Architecture
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections around Attention and MLP
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer

        # Token and Positional Embedding Tables
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])

        # Final LayerNorm and Language Model Head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share weights between input token embeddings and output linear head
        self.token_embeddings.weight = self.lm_head.weight

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            idx: LongTensor of shape (B, T)
            targets: Optional LongTensor of shape (B, T)
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross-entropy loss (if targets provided)
        """
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds maximum block size {self.block_size}"

        # Step 1: Form embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        tok_emb = self.token_embeddings(idx)  # shape (B, T, n_embd)
        pos_emb = self.position_embeddings(pos)  # shape (T, n_embd)
        x = self.drop(tok_emb + pos_emb)  # shape (B, T, n_embd)

        # Step 2: Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Step 3: Final LayerNorm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.lm_head(x)  # shape (B, T, vocab_size)

        # Step 4: Compute loss if targets are supplied
        loss = None
        if targets is not None:
            # Flatten (B, T, V) -> (B * T, V) and (B, T) -> (B * T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressively generates new tokens conditioned on an input context.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds maximum block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # Forward pass to get logits for the last token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Optional Top-K truncation
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx