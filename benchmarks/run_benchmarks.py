"""
Benchmark — 5 systems compared:

  1. Plain RAG + LLM                         (fixed k, PyTorch sim, HF API)
  2. RAG + LLM + Triton                      (fixed k, Triton sim, HF API)
  3. RAG + LLM + Adaptive                    (adaptive k, multi-hop, PyTorch sim, HF API)
  4. RAG + Triton + LLM + Adaptive           (adaptive k, multi-hop, Triton sim, HF API)
  5. RAG + Triton + LLM + Adaptive + SGLang  (full system — Triton sim, adaptive, SGLang prefix cache)

Metrics:
  Retrieval  — recall@5, mrr, ndcg@5, context_precision
  Generation — RAGAS faithfulness, answer_relevancy, context_precision, context_recall
  System     — avg_latency_ms, p95_latency_ms, retrieval_ms, generation_ms, ttft_ms, avg_hops

Run:
  # Start SGLang first for system 5:
  python -m sglang.launch_server --model Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 30000

  python benchmarks/run_benchmarks.py
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline            import setup, index_documents
from src.retrieval           import embed_query
from src.retrieval.retriever import retrieve
from src.agents              import analyze_query, run_retrieval
from src.agents.generator_agent import generate_hf, generate
from src.caching             import report as cache_report
from src.evaluation.ragas_eval  import evaluate_ragas
from src.evaluation.metrics     import all_retrieval_metrics

DOCS = [
    {"id": "doc1", "text": "Transformers use self-attention to process sequences in parallel. BERT uses bidirectional encoding; GPT uses causal left-to-right attention."},
    {"id": "doc2", "text": "RAG combines a dense retriever with a generative model. The retriever fetches relevant documents; the generator produces an answer conditioned on them."},
    {"id": "doc3", "text": "SGLang enables fast LLM inference via radix attention prefix caching. When the same document context appears repeatedly, SGLang reuses the KV cache, reducing time-to-first-token."},
    {"id": "doc4", "text": "Triton allows writing custom GPU kernels in Python. A fused cosine similarity kernel avoids two separate passes for normalization and dot product, improving GPU memory throughput."},
    {"id": "doc5", "text": "Vector databases store dense embeddings for semantic search. FAISS supports approximate nearest neighbor search, trading slight accuracy for much faster retrieval at scale."},
    {"id": "doc6", "text": "Multi-hop retrieval iteratively retrieves documents using previous results as context. Adaptive RAG adjusts the number of hops based on query complexity and retrieval confidence."},
]

QUERIES = [
    {"question": "What is the difference between BERT and GPT attention?",
     "relevant_ids": ["doc1_0"],
     "ground_truth": "BERT uses bidirectional attention; GPT uses causal left-to-right attention."},
    {"question": "How does RAG combine retrieval and generation?",
     "relevant_ids": ["doc2_0"],
     "ground_truth": "RAG retrieves relevant documents and conditions a generator on them to produce answers."},
    {"question": "How does SGLang prefix caching reduce time-to-first-token?",
     "relevant_ids": ["doc3_0"],
     "ground_truth": "SGLang reuses KV cache via radix attention when the same document prefix appears again."},
    {"question": "Why implement a custom Triton kernel for cosine similarity?",
     "relevant_ids": ["doc4_0"],
     "ground_truth": "A fused Triton kernel avoids two GPU passes for normalization and dot product, improving throughput."},
    {"question": "Explain multi-hop retrieval and how adaptive RAG decides the number of hops",
     "relevant_ids": ["doc6_0", "doc2_0"],
     "ground_truth": "Multi-hop retrieval fetches documents iteratively. Adaptive RAG decides hops based on query complexity and confidence."},
]


def run_and_collect(name, queries, run_fn):
    """Run all queries through run_fn, collect metrics + RAGAS samples."""
    retrieval_metrics = []
    latencies, retrieval_ms_list, gen_ms_list, ttft_list, hops_list = [], [], [], [], []
    ragas_samples = []

    print(f"\n--- {name} ---")
    for q in queries:
        result = run_fn(q["question"])

        retrieval_metrics.append(
            all_retrieval_metrics(result["retrieved_ids"], set(q["relevant_ids"]))
        )
        latencies.append(result["total_ms"])
        retrieval_ms_list.append(result.get("retrieval_ms", 0))
        gen_ms_list.append(result.get("generation_ms", 0))
        ttft_list.append(result.get("ttft_ms", 0))
        hops_list.append(result.get("hops", 1))

        ragas_samples.append({
            "question":     q["question"],
            "answer":       result["answer"],
            "contexts":     result["contexts"],
            "ground_truth": q["ground_truth"],
        })

    # Aggregate retrieval metrics
    agg = {k: round(float(np.mean([m[k] for m in retrieval_metrics])), 4)
           for k in retrieval_metrics[0]}

    # Real RAGAS
    print(f"  Running RAGAS evaluation...")
    ragas_scores = evaluate_ragas(ragas_samples)

    return {
        "system":            name,
        **agg,
        **{f"ragas_{k}": v for k, v in ragas_scores.items()},
        "avg_latency_ms":    round(float(np.mean(latencies)), 1),
        "p95_latency_ms":    round(float(np.percentile(latencies, 95)), 1),
        "avg_retrieval_ms":  round(float(np.mean(retrieval_ms_list)), 1),
        "avg_generation_ms": round(float(np.mean(gen_ms_list)), 1),
        "avg_ttft_ms":       round(float(np.mean(ttft_list)), 1),
        "avg_hops":          round(float(np.mean(hops_list)), 2),
    }


def main():
    index = setup()
    index_documents(index, DOCS)

    # ── System 1: Plain RAG + HF API ─────────────────────────────────────
    def system1(question):
        t0    = time.perf_counter()
        emb   = embed_query(question)
        chunks, _, _ = retrieve(index, emb, force_k=5, use_triton=False)
        gen   = generate_hf(question, chunks)
        total = (time.perf_counter() - t0) * 1000
        return {
            "retrieved_ids": [f"{c['chunk']['doc_id']}_{c['chunk']['chunk_id']}" for c in chunks],
            "answer":        gen["answer"],
            "contexts":      [c["chunk"]["text"] for c in chunks],
            "retrieval_ms":  0,
            "generation_ms": gen["total_ms"],
            "ttft_ms":       gen["ttft_ms"],
            "total_ms":      total,
            "hops":          1,
        }

    # ── System 2: RAG + Triton + HF API ──────────────────────────────────
    def system2(question):
        t0    = time.perf_counter()
        emb   = embed_query(question)
        chunks, _, _ = retrieve(index, emb, force_k=5, use_triton=True)
        gen   = generate_hf(question, chunks)
        total = (time.perf_counter() - t0) * 1000
        return {
            "retrieved_ids": [f"{c['chunk']['doc_id']}_{c['chunk']['chunk_id']}" for c in chunks],
            "answer":        gen["answer"],
            "contexts":      [c["chunk"]["text"] for c in chunks],
            "retrieval_ms":  0,
            "generation_ms": gen["total_ms"],
            "ttft_ms":       gen["ttft_ms"],
            "total_ms":      total,
            "hops":          1,
        }

    # ── System 3: RAG + Adaptive + HF API ────────────────────────────────
    def system3(question):
        t0       = time.perf_counter()
        analysis = analyze_query(question)
        ret      = run_retrieval(index, analysis, use_triton=False)
        gen      = generate_hf(question, ret["chunks"])
        total    = (time.perf_counter() - t0) * 1000
        return {
            "retrieved_ids": [f"{c['chunk']['doc_id']}_{c['chunk']['chunk_id']}" for c in ret["chunks"]],
            "answer":        gen["answer"],
            "contexts":      [c["chunk"]["text"] for c in ret["chunks"]],
            "retrieval_ms":  ret["latency_ms"],
            "generation_ms": gen["total_ms"],
            "ttft_ms":       gen["ttft_ms"],
            "total_ms":      total,
            "hops":          ret["hops"],
        }

    # ── System 4: RAG + Triton + Adaptive + HF API ───────────────────────
    def system4(question):
        t0       = time.perf_counter()
        analysis = analyze_query(question)
        ret      = run_retrieval(index, analysis, use_triton=True)
        gen      = generate_hf(question, ret["chunks"])
        total    = (time.perf_counter() - t0) * 1000
        return {
            "retrieved_ids": [f"{c['chunk']['doc_id']}_{c['chunk']['chunk_id']}" for c in ret["chunks"]],
            "answer":        gen["answer"],
            "contexts":      [c["chunk"]["text"] for c in ret["chunks"]],
            "retrieval_ms":  ret["latency_ms"],
            "generation_ms": gen["total_ms"],
            "ttft_ms":       gen["ttft_ms"],
            "total_ms":      total,
            "hops":          ret["hops"],
        }

    # ── System 5: RAG + Triton + Adaptive + SGLang ───────────────────────
    def system5(question):
        t0       = time.perf_counter()
        analysis = analyze_query(question)
        ret      = run_retrieval(index, analysis, use_triton=True)
        gen      = generate(question, ret["chunks"])  # SGLang with prefix caching

        # record TTFT for prefix cache report
        from src.caching import record as record_ttft
        if ret["chunks"]:
            record_ttft([c["chunk"]["text"] for c in ret["chunks"]], gen["ttft_ms"])

        total = (time.perf_counter() - t0) * 1000
        return {
            "retrieved_ids": [f"{c['chunk']['doc_id']}_{c['chunk']['chunk_id']}" for c in ret["chunks"]],
            "answer":        gen["answer"],
            "contexts":      [c["chunk"]["text"] for c in ret["chunks"]],
            "retrieval_ms":  ret["latency_ms"],
            "generation_ms": gen["total_ms"],
            "ttft_ms":       gen["ttft_ms"],
            "total_ms":      total,
            "hops":          ret["hops"],
        }

    # ── Run all systems ───────────────────────────────────────────────────
    reports = []
    reports.append(run_and_collect("S1: Plain RAG (HF API)",                    QUERIES, system1))
    reports.append(run_and_collect("S2: RAG + Triton (HF API)",                 QUERIES, system2))
    reports.append(run_and_collect("S3: RAG + Adaptive (HF API)",               QUERIES, system3))
    reports.append(run_and_collect("S4: RAG + Triton + Adaptive (HF API)",      QUERIES, system4))
    reports.append(run_and_collect("S5: RAG + Triton + Adaptive + SGLang Cache", QUERIES, system5))

    # ── Print + save results ──────────────────────────────────────────────
    df = pd.DataFrame(reports).set_index("system")
    Path("benchmark_results").mkdir(exist_ok=True)
    df.to_csv("benchmark_results/comparison.csv")

    print("\n\n=== BENCHMARK RESULTS ===")
    print(df.to_string())

    # ── Plot ──────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plot_cols = ["recall@5", "mrr", "ndcg@5",
                     "ragas_faithfulness", "ragas_answer_relevancy",
                     "avg_ttft_ms", "avg_latency_ms", "avg_hops"]
        plot_cols = [c for c in plot_cols if c in df.columns]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        colors = sns.color_palette("husl", len(df))

        for i, col in enumerate(plot_cols):
            ax = axes[i]
            bars = ax.bar(range(len(df)), df[col], color=colors)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels([f"S{j+1}" for j in range(len(df))], fontsize=9)
            ax.set_title(col, fontweight="bold", fontsize=10)
            for bar, val in zip(bars, df[col]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        # Legend
        legend_labels = [f"S{j+1}: {name}" for j, name in enumerate(df.index)]
        fig.legend(legend_labels, loc="lower center", ncol=3, fontsize=9,
                   bbox_to_anchor=(0.5, 0.01))

        plt.suptitle("RAG System Benchmark Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig("benchmark_results/comparison.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to benchmark_results/comparison.png")
    except ImportError:
        pass

    print("\n=== SGLang Prefix Cache Report ===")
    for k, v in cache_report().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
