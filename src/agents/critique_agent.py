def critique(query, chunks, complexity, min_chunks=2, base_threshold=0.5):
    if len(chunks) < min_chunks:
        return {"sufficient": False, "refined_query": f"more details about: {query}"}

    avg_score = sum(c["score"] for c in chunks[:5]) / min(5, len(chunks))
    threshold = base_threshold + 0.2 * complexity

    if avg_score < threshold:
        return {"sufficient": False, "refined_query": f"detailed explanation of {query}"}

    return {"sufficient": True, "refined_query": None}
