import torch
from sentence_transformers import SentenceTransformer

_model = None

def load_embedder(model_name="BAAI/bge-large-en-v1.5"):
    global _model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = SentenceTransformer(model_name, device=device)
    return _model

def embed(texts, normalize=True):
    return _model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=normalize,
        convert_to_tensor=True,
        show_progress_bar=len(texts) > 100,
    ).float()

def embed_query(query):
    # BGE models need this prefix for queries
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    return embed([prefixed])[0]
