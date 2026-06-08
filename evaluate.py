import json
import math
from collections import Counter
from inference import generate_code, InferenceConfig


#implementation
def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re
    return re.findall(r"\w+|[^\w\s]", text.lower())


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """
    Compute corpus-level BLEU score between hypothesis and reference strings.
    Returns a score between 0 and 1.
    """
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)

    if len(hyp_tokens) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    log_score = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = ngrams(hyp_tokens, n)
        ref_ngrams = ngrams(ref_tokens, n)

        matches = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = max(sum(hyp_ngrams.values()), 1)
        precision = matches / total

        if precision == 0:
            return 0.0
        log_score += math.log(precision) / max_n

    return bp * math.exp(log_score)


# Evaluation loop
def evaluate(val_path: str = "data/val.jsonl", cfg: InferenceConfig = None):
    if cfg is None:
        cfg = InferenceConfig(backend="groq")

    # Load validation examples
    examples = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"Evaluating on {len(examples)} examples...\n")

    bleu_scores = []
    exact_matches = 0

    for i, ex in enumerate(examples):
        instruction = ex["instruction"]
        code_input = ex.get("input", "")
        reference = ex["output"]

        result = generate_code(instruction, code_input, cfg)
        hypothesis = result.code.strip()
        reference_stripped = reference.strip()

        score = bleu_score(hypothesis, reference_stripped)
        bleu_scores.append(score)

        is_exact = hypothesis == reference_stripped
        if is_exact:
            exact_matches += 1

        print(f"[{i+1}/{len(examples)}] BLEU={score:.3f}  Exact={is_exact}")
        print(f"  Instruction: {instruction[:60]}...")
        print()

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    exact_ratio = exact_matches / len(examples) if examples else 0.0

    print("=" * 50)
    print(f"Average BLEU:   {avg_bleu:.4f}")
    print(f"Exact Match:    {exact_ratio:.4f}  ({exact_matches}/{len(examples)})")


if __name__ == "__main__":
    evaluate()
