import numpy as np
import pandas as pd
from pathlib import Path
from .metrics import all_retrieval_metrics, faithfulness, answer_relevancy


def run_system(name, queries, run_fn):
    """
    queries: list of dicts with keys: query, relevant_ids, reference_answer
    run_fn:  fn(query_str) -> dict with keys: retrieved_ids, answer, latency_ms, hops
    """
    retrieval_metrics = []
    faith_scores  = []
    relev_scores  = []
    latencies     = []
    hops_list     = []

    for q in queries:
        result = run_fn(q["query"])
        retrieval_metrics.append(
            all_retrieval_metrics(result["retrieved_ids"], set(q["relevant_ids"]))
        )
        faith_scores.append(faithfulness(result["answer"], [result["answer"]]))
        relev_scores.append(answer_relevancy(result["answer"], q["query"]))
        latencies.append(result["latency_ms"])
        hops_list.append(result.get("hops", 1))

    agg = {k: float(np.mean([m[k] for m in retrieval_metrics])) for k in retrieval_metrics[0]}
    agg["faithfulness"]     = float(np.mean(faith_scores))
    agg["answer_relevancy"] = float(np.mean(relev_scores))

    return {
        "system":          name,
        **agg,
        "avg_latency_ms":  round(float(np.mean(latencies)), 2),
        "p95_latency_ms":  round(float(np.percentile(latencies, 95)), 2),
        "avg_hops":        round(float(np.mean(hops_list)), 2),
    }


def compare_systems(reports, output_dir="benchmark_results"):
    Path(output_dir).mkdir(exist_ok=True)
    df = pd.DataFrame(reports).set_index("system")
    df.to_csv(f"{output_dir}/comparison.csv")
    print(df.to_string())
    return df


def plot_comparison(df, output_dir="benchmark_results"):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cols = ["recall@5", "mrr", "ndcg@5", "faithfulness", "avg_latency_ms", "p95_latency_ms"]
        cols = [c for c in cols if c in df.columns]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            ax = axes[i]
            ax.bar(df.index, df[col], color=sns.color_palette("husl", len(df)))
            ax.set_title(col.upper(), fontweight="bold")
            ax.tick_params(axis="x", rotation=15)

        plt.suptitle("Baseline vs Adaptive vs Adaptive+Cache", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison.png", dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_dir}/comparison.png")
    except ImportError:
        print("matplotlib/seaborn not installed — skipping plot")
