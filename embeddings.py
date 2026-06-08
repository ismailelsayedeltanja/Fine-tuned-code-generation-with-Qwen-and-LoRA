

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


# Embedding model
class CodeEmbedder:
    """
    Wraps a HuggingFace encoder model to produce fixed-size code embeddings.
    Uses mean pooling over the last hidden state.
    """

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        print(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Embedding model running on: {self.device}")

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """
        Encode a list of code strings into embeddings.
        Returns a numpy array of shape (len(texts), hidden_size).
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)

            # Mean pooling: average over all token positions
            # Shape: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
            count = attention_mask.sum(dim=1)
            batch_embeddings = (sum_embeddings / count).cpu().numpy()

            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience method for a single string."""
        return self.embed([text])[0]


# Cosine similarity search
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity score between two embedding vectors. Range: -1 to 1."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_most_similar(query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> tuple[int, float]:
    """
    Find the index and score of the most similar item in the corpus.
    Returns (index, score).
    """
    scores = corpus_embeddings @ query_embedding
    norms = np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
    similarities = scores / norms
    best_idx = int(np.argmax(similarities))
    return best_idx, float(similarities[best_idx])


# Save / load embeddings to disk
def save_embeddings(embeddings: np.ndarray, path: str):
    """Save embeddings as a .npy file."""
    np.save(path, embeddings)
    print(f"Saved {embeddings.shape[0]} embeddings to {path}")


def load_embeddings(path: str) -> np.ndarray:
    """Load embeddings from a .npy file."""
    embeddings = np.load(path)
    print(f"Loaded {embeddings.shape[0]} embeddings from {path}")
    return embeddings


# Demo
if __name__ == "__main__":
    embedder = CodeEmbedder()

    # A small code corpus
    corpus = [
        "def add(a, b):\n    return a + b",
        "def multiply(a, b):\n    return a * b",
        "def read_file(path):\n    with open(path) as f:\n        return f.read()",
        "def sort_list(items):\n    return sorted(items)",
        "def fetch_url(url):\n    import requests\n    return requests.get(url).text",
    ]

    print("Embedding corpus...")
    corpus_embeddings = embedder.embed(corpus)
    print(f"Corpus shape: {corpus_embeddings.shape}")  # (5, 768)

    # Query
    query = "function that opens and reads a text file"
    query_embedding = embedder.embed_one(query)

    idx, score = find_most_similar(query_embedding, corpus_embeddings)
    print(f"\nQuery: '{query}'")
    print(f"Best match (score={score:.3f}):\n{corpus[idx]}")

    # Save to disk
    save_embeddings(corpus_embeddings, "outputs/corpus_embeddings.npy")
