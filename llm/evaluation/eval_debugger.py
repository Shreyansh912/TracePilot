import os
import ast
import json
import torch
import torch.nn.functional as F

from llm.tokenizer.bpe_tokenizer import CodeBPETokenizer
from llm.model.transformer import MiniGPT


def generate_patch(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.4,
    top_k: int = 10,
    repetition_penalty: float = 1.3,
    block_size: int = 256,
    eos_token_id: int = None,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to block_size if sequence exceeds context length
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Apply repetition penalty to suppress token loops
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Top-K truncation
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Terminate immediately on <|eos|>
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

    return idx


def run_ast_verification(code: str) -> bool:
    try:
        ast.parse(code)
        print("[+] AST Syntax Check: PASSED (Valid Python AST)")
        return True
    except SyntaxError:
        pass

    # Fallback: check extracted valid code blocks
    lines = [
        line
        for line in code.split("\n")
        if any(
            line.strip().startswith(kw)
            for kw in ["def ", "return ", "if ", "raise ", "try:", "except ", "pass"]
        )
        or line.startswith("    ")
    ]
    if lines:
        try:
            ast.parse("\n".join(lines))
            print("[+] AST Syntax Check (Extracted Block): PASSED")
            return True
        except SyntaxError as e:
            print(f"[-] AST Syntax Check: FAILED ({e})")
            return False

    print("[!] Note: No recognizable Python block detected in output.")
    return False


def test_debugging_inference():
    checkpoint_path = os.path.join("llm", "checkpoints", "best_minigpt.pt")
    bpe_dir = os.path.join("llm", "tokenizer")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Evaluation on: {device}")
    print(f"[*] Loading checkpoint: {checkpoint_path}")

    tokenizer = CodeBPETokenizer(tokenizer_dir=bpe_dir)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    print(f"[*] Architecture Config: {config}")

    # Initialize MiniGPT with the exact checkpoint config
    model = MiniGPT(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Safely obtain EOS token ID via tokenizer.encode
    eos_tokens = tokenizer.encode("<|eos|>")
    eos_token_id = eos_tokens[0] if eos_tokens else None

    # Load first sample from dataset
    dataset_path = os.path.join(
        "datasets", "synthetic_debugging_dataset_scaled.jsonl"
    )
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            sample = json.loads(f.readline())
        buggy_code = sample["buggy_code"]
        error_msg = sample.get(
            "error_message", sample.get("error_trace", "Error")
        )
        ground_truth = sample.get("fixed_code", "")
    else:
        buggy_code = (
            "def calculate_ratio(total, count):\n    pass\n    return total / count"
        )
        error_msg = "ZeroDivisionError: division by zero"
        ground_truth = ""

    # Exact training prompt format
    prompt = f"<|code|>\n{buggy_code}\n<|error|>\n{error_msg}\n<|fix|>\n"

    prompt_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt_ids)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print("\n" + "=" * 60)
    print(
        f"INPUT PROMPT ({prompt_len} tokens, max block {config.get('block_size', 256)}):"
    )
    print(prompt.strip())
    print("=" * 60)

    # Generate fix
    out = generate_patch(
        model=model,
        idx=x,
        max_new_tokens=64,
        temperature=0.3,
        top_k=8,
        repetition_penalty=1.2,
        block_size=config.get("block_size", 256),
        eos_token_id=eos_token_id,
    )

    all_tokens = out[0].tolist()
    generated_tokens = all_tokens[prompt_len:]

    print(f"[*] Generated Token IDs ({len(generated_tokens)}): {generated_tokens}")

    raw_output = tokenizer.decode(generated_tokens)
    print("\n[MODEL RAW OUTPUT]:")
    print(repr(raw_output))

    clean_text = raw_output.replace("<|eos|>", "").strip()
    print("\n[FORMATTED PATCH COMPLETION]:")
    print(clean_text)
    print("=" * 60)

    if ground_truth:
        print("[GROUND TRUTH FIX]:")
        print(ground_truth.strip())
        print("=" * 60)

    print("\n[*] Running AST Syntax Verification...")
    run_ast_verification(clean_text)


if __name__ == "__main__":
    test_debugging_inference()