import torch
import numpy as np
from ..kernels.cosine_similarity import cosine_similarity


# Simple in-memory index: list of chunks + stacked embedding tensor
class VectorIndex:
    def __init__(self):
        self.chunks = []
        self.embeddings = None

    def add(self, chunks, embeddings):
        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)

    def search(self, query_emb, top_k, use_triton=True):
        if not self.chunks:
            return []
        sims  = cosine_similarity(query_emb.unsqueeze(0), self.embeddings, use_triton=use_triton)[0]
        k     = min(top_k, len(self.chunks))
        scores, idxs = torch.topk(sims, k)
        return [
            {"chunk": self.chunks[i.item()], "score": scores[r].item()}
            for r, i in enumerate(idxs)
        ]


def compute_entropy(scores):
    probs = torch.softmax(torch.tensor(scores) * 10, dim=0).numpy()
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def adaptive_k(entropy, complexity, min_k=2, max_k=20, low=0.3, high=0.7):
    if entropy < low:
        k = min_k
    elif entropy > high:
        k = max_k
    else:
        ratio = (entropy - low) / (high - low)
        k = int(min_k + ratio * (max_k - min_k))
    k = int(k * (1 + 0.5 * complexity))
    return max(min_k, min(max_k, k))


def retrieve(index, query_emb, complexity=0.5, threshold=0.4, force_k=None, use_triton=True):
    candidates = index.search(query_emb, top_k=20, use_triton=use_triton)
    if not candidates:
        return [], 0.0, 0

    scores  = [c["score"] for c in candidates]
    entropy = compute_entropy(scores)
    k       = force_k or adaptive_k(entropy, complexity)

    filtered = [c for c in candidates if c["score"] >= threshold][:k]
    if not filtered:
        filtered = candidates[:k]

    confidence = float(np.mean([c["score"] for c in filtered]))
    return filtered, entropy, k
