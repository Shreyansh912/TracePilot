"""
Formats JSONL debugging samples into tokenized sequences with target masking:
Pattern:
<|code|> {code} <|error|> {error} <|fix|> {explanation}\n{fix} <|eos|>

Target Masking:
Loss is calculated ONLY on the tokens following <|fix|>.
Tokens prior to <|fix|> receive target label = -100 (ignored in cross-entropy).
"""
import json
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer


class DebuggingDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: CodeBPETokenizer, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        code_tok = "<|code|>"
        err_tok = "<|error|>"
        fix_tok = "<|fix|>"
        eos_tok = "<|eos|>"
        pad_id = tokenizer.token_to_id("<|pad|>")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)

                # Format prompt portion (Conditioning context)
                prompt_text = f"{code_tok}\n{item['code']}\n{err_tok}\n{item['error']}\n{fix_tok}\n"
                # Format target completion portion (Supervised output)
                completion_text = f"{item['explanation']}\n{item['fix']}{eos_tok}"

                prompt_ids = tokenizer.encode(prompt_text)
                completion_ids = tokenizer.encode(completion_text)

                input_ids = prompt_ids + completion_ids
                # Mask prompt: Target is -100 for all prompt tokens
                labels = ([-100] * len(prompt_ids)) + completion_ids

                # Truncate to max context length if needed
                if len(input_ids) > max_seq_len:
                    input_ids = input_ids[:max_seq_len]
                    labels = labels[:max_seq_len]

                # Right-pad with <|pad|> token to keep uniform batch sizes
                pad_len = max_seq_len - len(input_ids)
                if pad_len > 0:
                    input_ids = input_ids + [pad_id] * pad_len
                    # Mask padding tokens so loss is never computed on padding
                    labels = labels + [-100] * pad_len

                self.samples.append((
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long)
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]