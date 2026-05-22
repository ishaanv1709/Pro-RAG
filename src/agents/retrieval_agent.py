import time
from ..retrieval.embedder  import embed_query
from ..retrieval.retriever import retrieve
from ..retrieval.reranker  import rerank
from ..adaptive.controller import get_policy, should_continue


def run_retrieval(index, analysis, latency_budget=2000, use_triton=True):
    t_start      = time.perf_counter()
    seen         = set()
    all_chunks   = []
    k_values     = []
    confidences  = []
    prev_conf    = 1.0

    queries = [analysis["query"]] + analysis["sub_queries"]

    for hop, query in enumerate(queries):
        elapsed = (time.perf_counter() - t_start) * 1000
        policy  = get_policy(analysis["complexity"], prev_conf, elapsed, latency_budget)

        query_emb              = embed_query(query)
        chunks, entropy, k     = retrieve(index, query_emb,
                                          complexity=analysis["complexity"],
                                          threshold=0.4,
                                          force_k=policy["top_k"],
                                          use_triton=use_triton)

        # deduplicate
        new_chunks = []
        for c in chunks:
            key = (c["chunk"]["doc_id"], c["chunk"]["chunk_id"])
            if key not in seen:
                seen.add(key)
                new_chunks.append(c)

        if policy["use_rerank"] and new_chunks:
            new_chunks = rerank(query, new_chunks, threshold=policy["rerank_threshold"])

        all_chunks.extend(new_chunks)
        confidence = float(sum(c["score"] for c in new_chunks) / len(new_chunks)) if new_chunks else 0.0
        confidences.append(confidence)
        k_values.append(k)
        prev_conf = confidence

        elapsed = (time.perf_counter() - t_start) * 1000
        if not should_continue(confidence, hop + 1, policy["n_hops"], elapsed, latency_budget):
            break

    return {
        "chunks":      all_chunks,
        "hops":        hop + 1,
        "k_values":    k_values,
        "confidences": confidences,
        "latency_ms":  (time.perf_counter() - t_start) * 1000,
    }
