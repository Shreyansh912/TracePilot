import os
import ast
import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer
from llm.model.transformer import MiniGPT


class TracePilotEngine:
    def __init__(
        self,
        checkpoint_path: str = "llm/checkpoints/best_debugger.pt",
        tokenizer_dir: str = "llm/tokenizer",
        device: Optional[str] = None
    ):
        self.device = torch.device(
            device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        print(f"[*] TracePilot Engine booting on: {self.device}")

        # 1. Load Tokenizer
        self.tokenizer = CodeBPETokenizer(tokenizer_dir=tokenizer_dir)
        eos_tokens = self.tokenizer.encode("<|eos|>")
        self.eos_token_id = eos_tokens[0] if eos_tokens else None

        # 2. Load Checkpoint
        assert os.path.exists(checkpoint_path), f"Checkpoint missing: {checkpoint_path}"
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.block_size = self.config.get("block_size", 256)

        # 3. Instantiate and warm up MiniGPT
        self.model = MiniGPT(**self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"[+] TracePilot Engine ready. Loaded {self.model.get_num_params():,} parameters.")

    @torch.no_grad()
    def generate_patch(
        self,
        buggy_code: str,
        error_message: str,
        max_new_tokens: int = 64,
        temperature: float = 0.3,
        top_k: int = 8,
        repetition_penalty: float = 1.2
    ) -> Dict[str, Any]:
        start_time = time.time()

        # Build prompt identical to training format
        prompt = f"<|code|>\n{buggy_code}\n<|error|>\n{error_message}\n<|fix|>\n"
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Autoregressive generation
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if self.eos_token_id is not None and idx_next.item() == self.eos_token_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

        latency_ms = (time.time() - start_time) * 1000
        gen_tokens = idx[0].tolist()[prompt_len:]
        raw_patch = self.tokenizer.decode(gen_tokens)
        clean_patch = raw_patch.replace("<|eos|>", "").strip()

        # Static AST Validation
        ast_valid = False
        try:
            ast.parse(clean_patch)
            ast_valid = True
        except SyntaxError:
            pass

        return {
            "patch": clean_patch,
            "ast_valid": ast_valid,
            "tokens_generated": len(gen_tokens),
            "latency_ms": round(latency_ms, 2)
        }