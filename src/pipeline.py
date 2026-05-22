import time
from .retrieval.embedder     import load_embedder, embed
from .retrieval.chunker      import chunk_text
from .retrieval.retriever    import VectorIndex
from .retrieval.reranker     import load_reranker
from .agents.query_analyzer  import analyze_query
from .agents.retrieval_agent import run_retrieval
from .agents.critique_agent  import critique
from .agents.generator_agent import generate as _generate
from .caching.prefix_cache   import record as record_ttft


def setup(config=None):
    cfg = config or {}
    load_embedder(cfg.get("embedding_model", "BAAI/bge-large-en-v1.5"))
    load_reranker()
    return VectorIndex()


def index_documents(index, documents, chunk_size=512, overlap=50):
    for doc in documents:
        chunks = chunk_text(doc["text"], doc["id"], size=chunk_size, overlap=overlap)
        embs   = embed([c["text"] for c in chunks])
        index.add(chunks, embs)
    print(f"Indexed {len(documents)} docs → {len(index.chunks)} chunks")


def query(index, question, latency_budget=2000):
    t_start  = time.perf_counter()
    analysis = analyze_query(question)

    retrieval = run_retrieval(index, analysis, latency_budget=latency_budget)

    # one extra hop if context is weak
    check = critique(question, retrieval["chunks"], analysis["complexity"])
    if not check["sufficient"] and check["refined_query"]:
        extra = run_retrieval(index, analyze_query(check["refined_query"]), latency_budget=latency_budget)
        retrieval["chunks"].extend(extra["chunks"])
        retrieval["hops"] += extra["hops"]

    gen = _generate(question, retrieval["chunks"])

    # record real TTFT so prefix cache tracker can compute speedup
    cache_hit = False
    if retrieval["chunks"]:
        doc_texts = [c["chunk"]["text"] for c in retrieval["chunks"]]
        cache_hit = record_ttft(doc_texts, gen["ttft_ms"])

    return {
        "question":       question,
        "answer":         gen["answer"],
        "contexts":       [c["chunk"]["text"] for c in retrieval["chunks"]],
        "retrieved_ids":  [f"{c['chunk']['doc_id']}_{c['chunk']['chunk_id']}" for c in retrieval["chunks"]],
        "num_chunks":     len(retrieval["chunks"]),
        "hops":           retrieval["hops"],
        "k_values":       retrieval["k_values"],
        "complexity":     analysis["complexity"],
        "confidence":     retrieval["confidences"][-1] if retrieval["confidences"] else 0.0,
        "ttft_ms":        gen["ttft_ms"],
        "generation_ms":  gen["total_ms"],
        "retrieval_ms":   retrieval["latency_ms"],
        "total_ms":       (time.perf_counter() - t_start) * 1000,
        "cache_hit":      cache_hit,
        "backend":        gen.get("backend", "sglang"),
    }
