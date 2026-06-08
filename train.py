import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from config import TrainingConfig


def load_data(data_path: str) -> Dataset:
    """Load training data from a JSONL file."""
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_prompt(example: dict) -> dict:
    """
    Format each example into an instruction-following prompt.
    Input fields: instruction, input (optional), output
    """
    instruction = example.get("instruction", "")
    code_input = example.get("input", "")
    output = example.get("output", "")

    if code_input:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{code_input}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return {"text": prompt}


def build_qlora_model(model_name: str, cfg: TrainingConfig):
    """Load model in 4-bit and attach LoRA adapters."""

    # 4-bit quantization config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for k-bit training (freezes base weights)
    model = prepare_model_for_kbit_training(model)

    # LoRA config — only trains a small set of adapter weights
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.lora_target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(cfg: TrainingConfig):
    """Main training loop."""

    # 1. Load and format data
    dataset = load_data(cfg.data_path)
    dataset = dataset.map(format_prompt)
    print(f"Dataset size: {len(dataset)} examples")

    # 2. Build model + tokenizer with QLoRA
    model, tokenizer = build_qlora_model(cfg.model_name, cfg)

    # 3. HuggingFace TrainingArguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        warmup_ratio=0.05,
    )

    # 4. SFTTrainer handles tokenization + packing automatically
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # 5. Save LoRA adapter weights only (not the full base model)
    adapter_path = os.path.join(cfg.output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to: {adapter_path}")


if __name__ == "__main__":
    cfg = TrainingConfig()
    train(cfg)
