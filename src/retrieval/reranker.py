_reranker = None

def load_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _reranker
    from sentence_transformers import CrossEncoder
    _reranker = CrossEncoder(model_name)

def rerank(query, chunks, threshold=0.3):
    if not chunks or _reranker is None:
        return chunks
    pairs  = [(query, c["chunk"]["text"]) for c in chunks]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    result = [c for c, s in ranked if s >= threshold]
    return result if result else [ranked[0][0]]
