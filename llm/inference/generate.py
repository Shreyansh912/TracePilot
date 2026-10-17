"""
llm/inference/generate.py
Inference engine with strict consecutive deduplication, frequency penalty,
and atomic special token mapping.
"""
import re
from typing import Optional, List
from collections import Counter
import torch
from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer
from llm.model.transformer import MiniGPT


class DebugGenerator:
    def __init__(self, model_path: str, bpe_dir: str, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CodeBPETokenizer(tokenizer_dir=bpe_dir)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        cfg = checkpoint["config"]
        
        self.model = MiniGPT(
            vocab_size=cfg["vocab_size"],
            block_size=cfg["block_size"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            dropout=0.0
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.special_tokens = ["<|code|>", "<|error|>", "<|fix|>", "<|eos|>"]
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")
        self.newline_id = self.tokenizer.token_to_id("\n")

    def encode_with_special_tokens(self, text: str) -> List[int]:
        pattern = f"({'|'.join(re.escape(tok) for tok in self.special_tokens)})"
        parts = re.split(pattern, text)
        
        token_ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                tid = self.tokenizer.token_to_id(part)
                if tid is not None:
                    token_ids.append(tid)
                else:
                    token_ids.extend(self.tokenizer.encode(part))
            else:
                token_ids.extend(self.tokenizer.encode(part))
        return token_ids

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        min_new_tokens: int = 12,
        **kwargs
    ) -> str:
        tokens = self.encode_with_special_tokens(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        generated_tokens: List[int] = []

        for step in range(max_new_tokens):
            x_cond = x if x.size(1) <= self.model.block_size else x[:, -self.model.block_size:]
            logits, _ = self.model(x_cond)
            logits = logits[:, -1, :].clone()

            # Prevent premature EOS before minimal tokens have been synthesized
            if step < min_new_tokens and self.eos_id is not None:
                logits[0, self.eos_id] = float("-inf")

            # 1. Hard Block: Never allow the exact same token to be selected twice in a row
            if generated_tokens:
                last_token = generated_tokens[-1]
                logits[0, last_token] = float("-inf")

            # 2. Additive frequency penalty on the last 20 generated tokens
            if generated_tokens:
                recent_window = generated_tokens[-20:]
                counts = Counter(recent_window)
                for tid, freq in counts.items():
                    logits[0, tid] -= (freq * 3.0)

            # Deterministic greedy argmax selection
            next_id = torch.argmax(logits, dim=-1).item()

            if self.eos_id is not None and next_id == self.eos_id and step >= min_new_tokens:
                break

            generated_tokens.append(next_id)
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            x = torch.cat((x, next_tensor), dim=1)

        decoded = self.tokenizer.decode(generated_tokens)
        return decoded