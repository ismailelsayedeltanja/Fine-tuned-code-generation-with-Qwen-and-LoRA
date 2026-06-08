import json
import requests
from pydantic import BaseModel, ValidationError
from config import InferenceConfig


# Pydantic schema — defines what valid output looks like

class CodeOutput(BaseModel):
    language: str         
    code: str              
    explanation: str       

    @classmethod
    def from_raw(cls, raw_text: str) -> "CodeOutput":
        """
        Parse the model output.
        The model is prompted to return JSON, so we try to parse it directly.
        Falls back to wrapping the raw text if JSON parsing fails.
        """
        # Strip markdown fences if present
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])  # remove first and last line

        try:
            data = json.loads(cleaned)
            return cls(**data)
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat the whole response as code
            return cls(
                language="python",
                code=raw_text.strip(),
                explanation="(no structured explanation provided)",
            )


# Backend implementations

class HuggingFaceBackend:
    """Load LoRA adapter on top of the base model and run local inference."""

    def __init__(self, cfg: InferenceConfig):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        print(f"Loading base model: {cfg.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model, trust_remote_code=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Loading LoRA adapter from: {cfg.adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, cfg.adapter_path)
        self.model.eval()
        self.cfg = cfg

    def generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class GroqBackend:
    """Use Groq cloud API for fast inference."""

    def __init__(self, cfg: InferenceConfig):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Run: pip install groq")

        import os
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set the GROQ_API_KEY environment variable.")

        self.client = Groq(api_key=api_key)
        self.cfg = cfg

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.cfg.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
        )
        return response.choices[0].message.content


class OllamaBackend:
    """Call a locally running Ollama server."""

    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.url = f"{cfg.ollama_base_url}/api/generate"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.cfg.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_predict": self.cfg.max_new_tokens,
            },
        }
        response = requests.post(self.url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"]


# Unified interface

def get_backend(cfg: InferenceConfig):
    if cfg.backend == "huggingface":
        return HuggingFaceBackend(cfg)
    elif cfg.backend == "groq":
        return GroqBackend(cfg)
    elif cfg.backend == "ollama":
        return OllamaBackend(cfg)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}. Choose huggingface / groq / ollama")


def build_prompt(instruction: str, code_input: str = "") -> str:
    """
    Build the prompt that asks the model to return structured JSON.
    This is the same format used during training.
    """
    json_schema = '{"language": "...", "code": "...", "explanation": "..."}'

    if code_input:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{code_input}\n\n"
            f"### Response:\nRespond ONLY with a JSON object matching this schema:\n"
            f"{json_schema}\n"
        )
    else:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\nRespond ONLY with a JSON object matching this schema:\n"
            f"{json_schema}\n"
        )


def generate_code(instruction: str, code_input: str = "", cfg: InferenceConfig = None) -> CodeOutput:
    """
    Main entry point.
    Returns a validated CodeOutput object.
    """
    if cfg is None:
        cfg = InferenceConfig()

    backend = get_backend(cfg)
    prompt = build_prompt(instruction, code_input)

    raw_output = backend.generate(prompt)

    try:
        result = CodeOutput.from_raw(raw_output)
    except ValidationError as e:
        print(f"Validation error: {e}")
        result = CodeOutput(language="unknown", code=raw_output, explanation="validation failed")

    return result


# Quick manual test

if __name__ == "__main__":
    cfg = InferenceConfig(backend="groq")   # change to "ollama" or "huggingface"

    result = generate_code(
        instruction="Write a Python function that reads a CSV file and returns a list of dictionaries.",
        cfg=cfg,
    )

    print("Language:", result.language)
    print("Explanation:", result.explanation)
    print("Code:\n", result.code)
