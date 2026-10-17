import os
import ast
import json
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer
from llm.model.transformer import MiniGPT


def generate_patch(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.3,
    top_k: int = 8,
    repetition_penalty: float = 1.2,
    block_size: int = 256,
    eos_token_id: int = None
) -> str:
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = model(idx_cond)
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

            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

    return idx


def is_valid_ast(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        pass

    lines = [
        line for line in code.split("\n")
        if any(line.strip().startswith(kw) for kw in ["def ", "return ", "if ", "raise ", "try:", "except ", "pass"])
        or line.startswith("    ")
    ]
    if lines:
        try:
            ast.parse("\n".join(lines))
            return True
        except SyntaxError:
            return False
    return False


def run_benchmark(sample_limit: int = 100):
    checkpoint_path = os.path.join("llm", "checkpoints", "best_minigpt.pt")
    bpe_dir = os.path.join("llm", "tokenizer")
    dataset_path = os.path.join("datasets", "synthetic_debugging_dataset_scaled.jsonl")

    assert os.path.exists(checkpoint_path), f"Checkpoint missing: {checkpoint_path}"
    assert os.path.exists(dataset_path), f"Dataset missing: {dataset_path}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running TracePilot Batch Benchmark on {device}")

    tokenizer = CodeBPETokenizer(tokenizer_dir=bpe_dir)
    eos_tokens = tokenizer.encode("<|eos|>")
    eos_token_id = eos_tokens[0] if eos_tokens else None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MiniGPT(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load dataset records
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Evaluate on the validation slice (last 15%)
    val_split_start = int(len(records) * 0.85)
    val_records = records[val_split_start:]
    if sample_limit and sample_limit < len(val_records):
        val_records = val_records[:sample_limit]

    print(f"[*] Evaluating on {len(val_records)} validation samples...\n")

    ast_valid_count = 0
    exact_match_count = 0
    latencies = []

    for item in tqdm(val_records, desc="Evaluating"):
        buggy_code = item["buggy_code"]
        error_msg = item.get("error_message", item.get("error_trace", "Error"))
        ground_truth = item.get("fixed_code", "").strip()

        prompt = f"<|code|>\n{buggy_code}\n<|error|>\n{error_msg}\n<|fix|>\n"
        prompt_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        start_t = time.time()
        out = generate_patch(
            model=model,
            idx=x,
            max_new_tokens=64,
            temperature=0.3,
            top_k=8,
            repetition_penalty=1.2,
            block_size=config.get("block_size", 256),
            eos_token_id=eos_token_id
        )
        latency = time.time() - start_t
        latencies.append(latency)

        generated_ids = out[0].tolist()[prompt_len:]
        raw_patch = tokenizer.decode(generated_ids)
        clean_patch = raw_patch.replace("<|eos|>", "").strip()

        if is_valid_ast(clean_patch):
            ast_valid_count += 1

        if clean_patch == ground_truth:
            exact_match_count += 1

    total = len(val_records)
    avg_latency = sum(latencies) / total if total > 0 else 0

    print("\n" + "=" * 50)
    print("           BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total Evaluated Samples : {total}")
    print(f"AST Syntactic Validity  : {ast_valid_count / total * 100:.2f}% ({ast_valid_count}/{total})")
    print(f"Exact Match (EM) Rate   : {exact_match_count / total * 100:.2f}% ({exact_match_count}/{total})")
    print(f"Avg Inference Latency   : {avg_latency * 1000:.2f} ms/sample")
    print("=" * 50)


if __name__ == "__main__":
    run_benchmark(sample_limit=100)