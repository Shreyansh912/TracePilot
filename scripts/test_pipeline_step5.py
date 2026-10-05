"""
Verifies BPE tokenization, compression efficiency over character tokenization,
and special debugging token ID resolution.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.tokenizer.char_tokenizer import CharTokenizer
from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer, SPECIAL_TOKENS


def test_bpe_pipeline():
    corpus_path = os.path.join("datasets", "raw", "python_code_corpus.txt")
    bpe_dir = os.path.join("llm", "tokenizer", "bpe_model")
    char_vocab_path = os.path.join("llm", "tokenizer", "vocab.json")

    print("[*] Initializing Step 5 Verification...")

    # 1. Train and save BPE tokenizer
    bpe = CodeBPETokenizer()
    bpe.train_from_files([corpus_path], vocab_size=600, min_frequency=2)
    bpe.save(bpe_dir)
    print(f"[+] BPE Tokenizer trained and saved to {bpe_dir} (Vocab size: {bpe.vocab_size})")

    # 2. Reload to ensure persistence integrity
    reloaded_bpe = CodeBPETokenizer(tokenizer_dir=bpe_dir)
    assert reloaded_bpe.vocab_size == bpe.vocab_size, "Reloaded vocab size mismatch!"
    print("[+] Model reload verified.")

    # 3. Check special tokens resolution
    print("\n[*] Special Debugging Tokens Check:")
    for token in SPECIAL_TOKENS:
        t_id = reloaded_bpe.token_to_id(token)
        print(f"    Token: {token:<12} -> ID: {t_id}")
        assert t_id is not None, f"Special token {token} failed to register in vocab!"

    # 4. Measure Token Compression Ratio vs Character Tokenization
    test_code = """
def quicksort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
""".strip()

    char_tok = CharTokenizer.load(char_vocab_path)
    # Character tokenizer may encounter characters not in tiny_code.txt, so count raw chars
    char_count = len(test_code)
    bpe_tokens = reloaded_bpe.encode(test_code)
    bpe_count = len(bpe_tokens)

    compression_ratio = char_count / bpe_count
    print(f"\n[*] Sequence Length Comparison for Function (quicksort):")
    print(f"    Character-level length: {char_count} tokens")
    print(f"    Subword BPE length:     {bpe_count} tokens")
    print(f"    Compression Ratio:      {compression_ratio:.2f}x reduction")

    assert compression_ratio > 2.0, "BPE tokenizer failed to achieve at least 2x compression!"
    print("\n[SUCCESS] Step 5: Subword BPE tokenizer operational with verified compression.")


if __name__ == "__main__":
    test_bpe_pipeline()