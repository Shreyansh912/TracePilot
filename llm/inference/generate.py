"""
llm/inference/generate.py
Inference engine with repetition penalty, temperature scaling, top-k sampling,
consecutive n-gram early stopping, and delimiter boundaries.
"""
from typing import List, Optional
import torch
import torch.nn.functional as F
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
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.3,
        top_k: int = 5,
        repetition_penalty: float = 1.45
    ) -> str:
        tokens = self.tokenizer.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)

        generated_tokens = []

        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= self.model.block_size else x[:, -self.model.block_size:]
            logits, _ = self.model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Apply repetition penalty across all tokens generated in current sequence
            if repetition_penalty != 1.0:
                for token_id in set(x[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Top-K Sampling
            if top_k > 0:
                val, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < val[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_id = next_token.item()

            # Stop conditions: EOS token reached
            if next_id == self.eos_id:
                break

            # Consecutive Repetition Guard: Stop if the same token repeats 3 times in a row
            generated_tokens.append(next_id)
            if len(generated_tokens) >= 3 and generated_tokens[-1] == generated_tokens[-2] == generated_tokens[-3]:
                # Discard repeating tail and break immediately
                generated_tokens = generated_tokens[:-2]
                break

            x = torch.cat((x, next_token), dim=1)

        return self.tokenizer.decode(generated_tokens)