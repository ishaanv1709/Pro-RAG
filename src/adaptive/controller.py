def get_policy(complexity, prev_confidence=1.0, elapsed_ms=0.0, latency_budget=2000):
    tight_budget = (latency_budget - elapsed_ms) < 500

    top_k = max(5, int(2 + complexity * 18))
    if tight_budget:
        top_k = max(5, top_k // 2)

    chunk_size = int(512 * (1.0 - 0.4 * complexity))
    chunk_size = max(128, min(2048, chunk_size))

    # be more permissive when previous retrieval was not confident
    rerank_threshold = 0.3 if prev_confidence < 0.5 else 0.5

    n_hops     = 2 if complexity > 0.6 else 1
    use_rerank = not tight_budget

    return {
        "top_k":            top_k,
        "chunk_size":       chunk_size,
        "rerank_threshold": rerank_threshold,
        "n_hops":           n_hops,
        "use_rerank":       use_rerank,
    }


def should_continue(confidence, hop, max_hops, elapsed_ms, latency_budget=2000, cutoff=0.75):
    if hop >= max_hops:
        return False
    if elapsed_ms > latency_budget * 0.9:
        return False
    if confidence >= cutoff:
        return False
    return True
