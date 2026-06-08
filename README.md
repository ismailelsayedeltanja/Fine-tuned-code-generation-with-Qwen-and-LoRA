# Code Generation Fine-Tuning

Fine-tune large language models for domain-specific code generation using **LoRA** and **QLoRA** techniques.
Supports multiple inference backends and includes structured output validation with Pydantic.

## Pipelne
-pip install -r requirements.txt
-python prepare_data.py      
-python train.py             
-python inference.py        
-python evaluate.py          

## Features

- **QLoRA fine-tuning** — 4-bit quantization + LoRA adapters, runs on a single consumer GPU
- **Multi-backend inference** — HuggingFace (local), Groq (cloud), Ollama (local server)
- **Structured output** — Pydantic validation ensures every response has `language`, `code`, and `explanation`
- **Code embeddings** — semantic similarity search over code using `microsoft/codebert-base`
- **Evaluation** — BLEU score and exact match metrics on a held-out validation set

---

## Project Structure

```
 requirements.txt
 1.config.py         > All hyperparameters and settings in one place
 2.train.py          > QLoRA fine-tuning loop
 3.inference.py      > Multi-backend inference + Pydantic validation
 4.embeddings.py     > Code embeddings with HuggingFace Transformers
 5.prepare_data.py   > Convert raw examples to JSONL training format
 6.evaluate.py       > BLEU + exact match evaluation
 7.data/             > Training and validation JSONL files (git-ignored)
 8.outputs/          > Checkpoints and embeddings (git-ignored)
```

---

## Quick Start

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Prepare training data

Edit the `EXAMPLES` list in `prepare_data.py` with your own code examples, then run:

```
python prepare_data.py
```

Each example has three fields:
```json
{
  "instruction": "Write a function that...",
  "input": "(optional) existing code or context",
  "output": "def my_function():\n    ..."
}
```

### 3. Configure the run

Open `config.py` and set your model and hyperparameters:

```python
@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    lora_r: int = 16
    num_epochs: int = 3
    batch_size: int = 2
```

### 4. Fine-tune

```bash
python train.py
```

The LoRA adapter is saved to `outputs/checkpoints/lora_adapter/`.

### 5. Run inference

**Option A — HuggingFace (local, uses the fine-tuned adapter):**
```bash
# Set backend="huggingface" in InferenceConfig
python inference.py
```

**Option B — Groq (cloud API):**
```bash
export GROQ_API_KEY=your_key_here
# Set backend="groq" in InferenceConfig
python inference.py
```

**Option C — Ollama (local server):**
```bash
ollama pull qwen2.5-coder:7b
ollama serve
# Set backend="ollama" in InferenceConfig
python inference.py
```

### 6. Generate code in your own script

```python
from inference import generate_code, InferenceConfig

cfg = InferenceConfig(backend="groq")

result = generate_code(
    instruction="Write a Python function that sorts a list of dictionaries by a given key.",
    cfg=cfg,
)

print(result.language)       >"python"
print(result.explanation)    >"Sorts a list of dicts using the sorted() built-in..."
print(result.code)           > def sort_by_key(items, key): ...
```

### 7. Generate code embeddings

```python
from embeddings import CodeEmbedder, find_most_similar

embedder = CodeEmbedder()  # loads microsoft/codebert-base

corpus = [
    "def add(a, b): return a + b",
    "def read_file(path): ...",
]

corpus_embeddings = embedder.embed(corpus)
query_embedding = embedder.embed_one("function that opens a file")

idx, score = find_most_similar(query_embedding, corpus_embeddings)
print(corpus[idx])   # def read_file(path): ...
```

### 8. Evaluate

```
python evaluate.py
```

---

## Supported Models

Any HuggingFace causal LM works. Tested with:

 Model Size  Notes 

 `Qwen/Qwen2.5-Coder-1.5B-Instruct`  1.5B  Fast, low VRAM 
 `Qwen/Qwen2.5-Coder-7B-Instruct` 7B  Better quality 
 `bigcode/starcoder2-3b`  3B  General code 

---

## Hardware Requirements

 Mode  Minimum GPU VRAM 

 QLoRA (4-bit) training  8 GB 
 Inference (HuggingFace, fp16)  8 GB 
 Groq / Ollama backends  No GPU needed 

---

## How QLoRA Works

1. The base model weights are loaded in **4-bit** precision (uses ~4x less memory).
2. **LoRA adapters** are small matrices injected into the attention layers. Only these are trained.
3. After training, you save only the adapter (~50 MB), not the full model.
4. At inference time, load the base model + adapter together.


