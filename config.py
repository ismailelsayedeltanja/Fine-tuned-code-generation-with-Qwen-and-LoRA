"""
config.py
All hyperparameters and paths in one place.
Change these values to switch models or datasets.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    # ---- Model ----
    # Any HuggingFace model ID works here:
    #   "Qwen/Qwen2.5-Coder-1.5B-Instruct"   (small, fast)
    #   "Qwen/Qwen2.5-Coder-7B-Instruct"      (better quality)
    #   "bigcode/starcoder2-3b"               (open code model)
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    # ---- Data ----
    data_path: str = "data/train.jsonl"

    # ---- Output ----
    output_dir: str = "outputs/checkpoints"

    # ---- LoRA ----
    # r = rank of the low-rank matrices (higher = more capacity, more memory)
    lora_r: int = 16
    lora_alpha: int = 32           # scaling factor (usually 2 * r)
    lora_dropout: float = 0.05

    # Which layers to inject LoRA into.
    # For Qwen models these are the attention projection layers.
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # ---- Training ----
    num_epochs: int = 3
    batch_size: int = 2            # keep small for GPU memory
    gradient_accumulation_steps: int = 8   # effective batch = batch_size * steps
    learning_rate: float = 2e-4
    max_seq_length: int = 1024


@dataclass
class InferenceConfig:
    # Path to the saved LoRA adapter (output of training)
    adapter_path: str = "outputs/checkpoints/lora_adapter"

    # The same base model used during training
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.2       # low = more deterministic code
    top_p: float = 0.95

    # Backend: "huggingface" | "groq" | "ollama"
    backend: str = "huggingface"

    # Groq settings (only used when backend = "groq")
    groq_model: str = "qwen-qwq-32b"

    # Ollama settings (only used when backend = "ollama")
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"
