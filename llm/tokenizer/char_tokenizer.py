"""
A deterministic, character-level tokenizer for small language models.
Maps unique characters to discrete integer IDs and back.
"""
from typing import List, Dict
import json
import os


class CharTokenizer:
    def __init__(self, chars: List[str] = None):
        """
        Initializes the character-level vocabulary.
        If chars is provided, constructs the mappings.
        """
        self.chars: List[str] = sorted(list(set(chars))) if chars else []
        self.vocab_size: int = len(self.chars)
        self.char_to_id: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    def fit(self, text: str) -> None:
        """
        Builds the vocabulary from an input corpus.
        """
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string into a list of integer token IDs.
        """
        try:
            return [self.char_to_id[ch] for ch in text]
        except KeyError as e:
            missing_char = e.args[0]
            raise ValueError(f"Character {repr(missing_char)} not found in tokenizer vocabulary.")

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of integer token IDs back into a string.
        """
        return "".join([self.id_to_char[i] for i in token_ids])

    def save(self, filepath: str) -> None:
        """
        Persists the vocabulary to a JSON file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "chars": self.chars
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        """
        Loads a tokenizer vocabulary from a JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(chars=data["chars"])
        return tokenizer