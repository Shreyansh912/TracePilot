"""
llm/tokenizer/bpe_tokenizer.py
Byte-Pair Encoding (BPE) subword tokenizer for TracePilot code modeling.
Features:
- Handles arbitrary unicode and bytes safely without OOV.
- Supports domain-specific debugging special tokens.
- Encodes strings to PyTorch tensors and decodes back cleanly.
"""
import os
from typing import List, Optional
from tokenizers import ByteLevelBPETokenizer


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|code|>",
    "<|error|>",
    "<|fix|>",
    "<|unk|>"
]


class CodeBPETokenizer:
    def __init__(self, tokenizer_dir: Optional[str] = None):
        """
        Initializes or loads a Byte-Level BPE Tokenizer.
        """
        if tokenizer_dir and os.path.exists(os.path.join(tokenizer_dir, "vocab.json")):
            self.tokenizer = ByteLevelBPETokenizer(
                os.path.join(tokenizer_dir, "vocab.json"),
                os.path.join(tokenizer_dir, "merges.txt")
            )
        else:
            self.tokenizer = ByteLevelBPETokenizer()

        self.pad_token = "<|pad|>"
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def train_from_files(
        self,
        files: List[str],
        vocab_size: int = 1000,
        min_frequency: int = 2
    ) -> None:
        """
        Trains the BPE tokenizer on a list of corpus text files.
        """
        self.tokenizer.train(
            files=files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS
        )

    def encode(self, text: str) -> List[int]:
        """Encodes text into a list of integer token IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes integer token IDs back to a string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(token_id)

    def save(self, output_dir: str) -> None:
        """Saves vocab.json and merges.txt to directory."""
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save_model(output_dir)


if __name__ == "__main__":
    # Smoke test when executed directly
    corpus_file = os.path.join("datasets", "raw", "python_code_corpus.txt")
    save_path = os.path.join("llm", "tokenizer", "bpe_model")

    print("[*] Training Byte-Level BPE Tokenizer...")
    bpe = CodeBPETokenizer()
    bpe.train_from_files([corpus_file], vocab_size=500, min_frequency=2)
    bpe.save(save_path)
    print(f"[+] BPE Model saved. Vocabulary Size: {bpe.vocab_size}")

    sample_code = "def binary_search(arr: List[int], target: int) -> int:"
    encoded = bpe.encode(sample_code)
    decoded = bpe.decode(encoded)
    print(f"\n[*] Sample Code: {repr(sample_code)}")
    print(f"[*] Token Count: {len(encoded)} tokens (IDs: {encoded[:8]}...)")
    print(f"[*] Decoded:     {repr(decoded)}")
    assert sample_code == decoded, "Decoded text did not match original input!"
    print("[SUCCESS] Subword BPE Tokenizer is fully operational.")