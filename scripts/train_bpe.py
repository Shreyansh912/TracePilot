"""
scripts/train_bpe.py
Trains and saves vocab.json and merges.txt in llm/tokenizer/
"""
import os
import json
from tokenizers import ByteLevelBPETokenizer

# Ensure directories exist
save_dir = os.path.join("llm", "tokenizer")
os.makedirs(save_dir, exist_ok=True)

# 1. Prepare raw text corpus from dataset for tokenizer training
dataset_path = os.path.join("datasets", "synthetic_debugging_dataset_scaled.jsonl")
corpus_temp_path = os.path.join("datasets", "temp_corpus.txt")

with open(dataset_path, "r", encoding="utf-8") as src, open(corpus_temp_path, "w", encoding="utf-8") as dst:
    for line in src:
        if line.strip():
            record = json.loads(line)
            dst.write(f"{record['buggy_code']}\n{record['error_message']}\n{record['fixed_code']}\n")

# 2. Train Byte-Level BPE Tokenizer
tokenizer = ByteLevelBPETokenizer()
special_tokens = ["<|pad|>", "<|eos|>", "<|code|>", "<|error|>", "<|fix|>", "<|unk|>"]

tokenizer.train(
    files=[corpus_temp_path],
    vocab_size=2000,
    min_frequency=2,
    special_tokens=special_tokens
)

# 3. Save vocab.json and merges.txt
tokenizer.save_model(save_dir)
print(f"[+] BPE vocab.json and merges.txt successfully saved to: {save_dir}")

# Cleanup
if os.path.exists(corpus_temp_path):
    os.remove(corpus_temp_path)