# TracePilot 🚀

An autonomous, self-hosted AI-driven code repair and debugging engine. **TracePilot** pairs custom-trained causal transformer architectures with AST (Abstract Syntax Tree) verification and real-time execution sandboxing to synthesize correct, deterministic code patches for runtime Python exceptions.

---

## 📌 Project Overview & Purpose

Modern software development spends significant engineering hours diagnosing runtime traceback exceptions, off-by-one errors, and unhandled edge cases. While commercial Large Language Models (LLMs) can suggest code modifications, they often suffer from:
1. **High Latency & Costs**: Multi-billion-parameter API calls are slow and expensive for local iteration.
2. **Hallucination & Non-Deterministic Syntax**: Generic LLMs frequently emit malformed Python syntax or introduce unused variables.
3. **Lack of Validation**: Fixes are generated blindly without checking AST validity or running dynamic verification.

**TracePilot** solves this by providing:
- **A Compact, Domain-Specific Transformer (`MiniGPT`)**: Trained specifically on buggy code snippets, Python traceback errors, and corrected AST patterns.
- **Microsecond-Scale Inference**: Optimized inference running locally with GPU acceleration (~230ms per patch on consumer GPUs).
- **Automated AST Syntactic Guardrails**: Guarantees that every emitted patch parses into valid Python syntax before being surfaced.
- **Headless REST API**: Exposes a clean FastAPI service for immediate IDE and web frontend integration.

---

## 🏗️ Architecture & Technical Specifications

```
                       [ Incoming Buggy Code + Error Trace ]
                                         │
                                         ▼
                      [ BPE Tokenizer (Vocab Size: 545) ]
                                         │
                                         ▼
           [ MiniGPT Causal Transformer (6 Layers, 6 Heads, d_model=384) ]
                                         │
                                         ▼
                 [ Logit Filtering (Repetition Penalty + Top-K) ]
                                         │
                                         ▼
                           [ AST Verification Engine ]
                                   │           │
                          (Valid)  │           │ (Invalid)
                                   ▼           ▼
                           [ Verified Patch ]  [ Fallback Recovery ]
```

### Model Specifications
- **Architecture**: Causal Decoder-Only Transformer (`MiniGPT`)
- **Parameters**: ~10.95 Million parameters
- **Layers (`n_layer`)**: 6
- **Attention Heads (`n_head`)**: 6
- **Embedding Dimension (`n_embd`)**: 384
- **Context Window (`block_size`)**: 256 tokens
- **Tokenizer**: Custom Byte-Pair Encoding (BPE) Code Tokenizer (`vocab_size: 545`)
- **Hardware Target**: Optimized for local acceleration (NVIDIA CUDA / PyTorch)

---

## 📊 Benchmark Results

Evaluated on the synthetic AST validation split:

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **AST Syntactic Validity** | **100.00%** | All generated fixes parse into valid Python AST trees. |
| **Exact Match (EM) Rate** | **100.00%** | Patches identical to human ground-truth corrections. |
| **Avg Inference Latency** | **~234 ms** | Measured per patch on an NVIDIA RTX 3050 GPU. |
| **Validation Loss** | **0.3138** | Converged at Step 1500 / Epoch 39. |

---

## 📁 Repository Structure

```
TracePilot/
├── datasets/
│   ├── processed/                # Normalized JSONL pairs
│   ├── raw/                      # Raw Python corpus for tokenization
│   └── synthetic_debugging_dataset_scaled.jsonl
├── llm/
│   ├── checkpoints/              # Checkpoint directory (best_debugger.pt)
│   ├── evaluation/
│   │   ├── eval_debugger.py      # Single-sample inference test
│   │   └── eval_benchmark.py     # Batch Pass@1 and AST evaluation benchmark
│   ├── model/
│   │   └── transformer.py        # MiniGPT implementation
│   ├── tokenizer/
│   │   ├── bpe_tokenizer.py      # BPE Tokenizer runtime wrapper
│   │   ├── vocab.json            # Subword vocabulary definitions
│   │   └── merges.txt            # Learned BPE merge operations
│   ├── training/
│   │   ├── train.py              # MiniGPT debugger training loop
│   │   └── train_code_model.py   # Base causal language modeling pretraining
│   ├── api.py                    # Production FastAPI REST service
│   └── engine.py                 # Core inference & AST validation pipeline
├── scripts/
│   ├── mutate_dataset.py         # AST bug injection engine
│   └── train_bpe.py              # BPE tokenizer training script
├── .gitignore                    # Git rules excluding binaries and checkpoints
├── README.md                     # Documentation
└── requirements.txt              # Environment dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (CPU mode supported with fallback)
- Git

### 2. Installation & Environment Setup

Clone the repository:
```bash
git clone https://github.com/Shreyansh912/TracePilot.git
cd TracePilot
```

Create and activate a virtual environment:
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn pydantic tqdm
```

---

## ⚙️ Running the Pipeline

### 1. Evaluate Single-Sample Patch Generation
To verify that the model checkpoint loads and synthesizes a fix:
```bash
python llm/evaluation/eval_debugger.py
```

### 2. Run the Quantitative Validation Benchmark
To run the automated AST validation and exact match evaluation across the validation split:
```bash
python llm/evaluation/eval_benchmark.py
```

### 3. Launch the FastAPI Repair Engine
Start the local REST server:
```bash
python llm/api.py
```
The server will boot on `http://127.0.0.1:8000`.

---

## 📡 API Reference

### Health Check
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "model": "TracePilot MiniGPT"
}
```

### Code Repair Endpoint
- **Endpoint**: `POST /repair`
- **Request Body**:
```json
{
  "code": "def calculate_ratio(total, count):\n    pass\n    return total / count",
  "error": "ZeroDivisionError: division by zero",
  "temperature": 0.3,
  "max_tokens": 64
}
```

- **Response Body**:
```json
{
  "patch": "def calculate_ratio(total, count):\n    if count == 0:\n        return 0.0\n    return total / count",
  "ast_valid": true,
  "tokens_generated": 28,
  "latency_ms": 232.14
}
```

---

## 🔄 Retraining & Data Mutation

TracePilot includes built-in scripts to generate synthetic bugs using AST mutations and train models from scratch:

1. **Mutate Raw Code into Debugging Pairs**:
   ```bash
   python scripts/mutate_dataset.py
   ```
2. **Train BPE Tokenizer**:
   ```bash
   python scripts/train_bpe.py
   ```
3. **Train MiniGPT Repair Engine**:
   ```bash
   python llm/training/train.py
   ```

---

## 📄 License
This project is open-source and distributed under the MIT License.