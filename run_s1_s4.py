"""
Run S1-S4 benchmark only — use when S5 chat session is already done.

Reads questions from benchmark_results/session_questions.json (saved by main.py).
Merges results with existing comparison.csv if present (to keep S5 row).

Run:
    python run_s1_s4.py
"""

import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path

from src.pipeline    import setup, index_documents
from src.loaders     import load_file
from src.retrieval   import embed_query
from src.retrieval.retriever import retrieve
from src.agents      import analyze_query, run_retrieval
from src.agents.generator_agent import generate
from src.evaluation.ragas_eval  import evaluate_ragas, evaluate_ragas_no_gt
from main import BENCHMARK_QA, _pack

PDF_PATH = "harry_potter_ch1.pdf"

def run_s1(index, question):
    t0 = time.perf_counter()
    chunks, _, _ = retrieve(index, embed_query(question), force_k=5, use_triton=False)
    return _pack(chunks, generate(question, chunks), t0, hops=1)

def run_s2(index, question):
    t0  = time.perf_counter()
    ret = run_retrieval(index, analyze_query(question), use_triton=False)
    return _pack(ret["chunks"], generate(question, ret["chunks"]), t0, ret["hops"])

def run_s3(index, question):
    t0 = time.perf_counter()
    chunks, _, _ = retrieve(index, embed_query(question), force_k=5, use_triton=True)
    return _pack(chunks, generate(question, chunks), t0, hops=1)

def run_s4(index, question):
    t0  = time.perf_counter()
    ret = run_retrieval(index, analyze_query(question), use_triton=True)
    return _pack(ret["chunks"], generate(question, ret["chunks"]), t0, ret["hops"])

SYSTEMS = [
    ("S1: Plain RAG",               run_s1),
    ("S2: RAG + Adaptive",          run_s2),
    ("S3: RAG + Triton",            run_s3),
    ("S4: RAG + Triton + Adaptive", run_s4),
]


def sep(char="─", n=64):
    print(char * n)


def find_gold_label(question):
    q = question.lower().strip()
    for qa in BENCHMARK_QA:
        if qa["question"].lower().strip() == q:
            return qa["ground_truth"]
    return None


def aggregate(results, name):
    ragas_samples = []
    for r in results:
        s = {"question": r["question"], "answer": r["answer"], "contexts": r["contexts"]}
        gt = find_gold_label(r["question"])
        if gt:
            s["ground_truth"] = gt
        ragas_samples.append(s)

    has_gt = all("ground_truth" in s for s in ragas_samples)
    label  = "4-metric" if has_gt else "2-metric (no ground truth)"
    print(f"    RAGAS ({label})...", end="", flush=True)
    try:
        scores = evaluate_ragas(ragas_samples) if has_gt else evaluate_ragas_no_gt(ragas_samples)
    except Exception as e:
        print(f" error: {e}")
        scores = {}
    print(" done")

    ttfts  = [r["ttft_ms"]  for r in results]
    totals = [r["total_ms"] for r in results]
    hops   = [r["hops"]     for r in results]
    return {
        "system":       name,
        **scores,
        "avg_ttft_ms":  round(np.mean(ttfts),           1),
        "p95_ttft_ms":  round(np.percentile(ttfts, 95), 1),
        "avg_total_ms": round(np.mean(totals),           1),
        "avg_hops":     round(np.mean(hops),             2),
    }


def main():
    questions_path = Path("benchmark_results/session_questions.json")
    csv_path       = Path("benchmark_results/comparison.csv")

    if questions_path.exists():
        with open(questions_path) as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions from session_questions.json")
    else:
        questions = [qa["question"] for qa in BENCHMARK_QA]
        print(f"session_questions.json not found — using {len(questions)} questions from BENCHMARK_QA")

    sep("═")
    print(f"S1-S4 BENCHMARK  —  {len(questions)} questions  —  S1-S5: SGLang local — only retrieval differs")
    sep("═")

    path = Path(PDF_PATH)
    if not path.exists():
        print(f"PDF not found: '{PDF_PATH}'")
        sys.exit(1)

    print(f"\nLoading {path.name}...")
    docs  = load_file(str(path))
    print("Setting up pipeline...")
    index = setup()
    index_documents(index, docs)
    print()

    reports = []

    for name, run_fn in SYSTEMS:
        print(f"\n  {name}")
        results = []
        for q in questions:
            try:
                r = run_fn(index, q)
                r["question"] = q
                results.append(r)
                print(f"  ✓ {q[:60]}{'...' if len(q) > 60 else ''}")
                print(f"    TTFT {r['ttft_ms']:.0f}ms | Total {r['total_ms']:.0f}ms | Hops {r['hops']}")
            except Exception as e:
                print(f"  ✗ Error on '{q[:50]}': {e}")
        if results:
            ttfts  = [r["ttft_ms"]  for r in results]
            totals = [r["total_ms"] for r in results]
            hops   = [r["hops"]     for r in results]
            sep()
            print(f"  {name} — {len(results)}/{len(questions)} questions completed")
            print(f"  Avg TTFT:   {np.mean(ttfts):.0f}ms  (p95 {np.percentile(ttfts, 95):.0f}ms)")
            print(f"  Avg Total:  {np.mean(totals):.0f}ms  (p95 {np.percentile(totals, 95):.0f}ms)")
            print(f"  Avg Hops:   {np.mean(hops):.1f}")
            sep()
            reports.append(aggregate(results, name))

    if not reports:
        print("\nNo results — is SGLang running on port 30000?")
        sys.exit(1)

    new_df = pd.DataFrame(reports).set_index("system")

    # Merge with existing S5 (and S6 if present) rows
    if csv_path.exists():
        existing = pd.read_csv(csv_path, index_col="system")
        # Drop old S1-S4 rows so we replace them cleanly
        existing = existing[~existing.index.str.match(r"S[1-4]:")]
        combined = pd.concat([new_df, existing]).sort_index()
    else:
        combined = new_df

    Path("benchmark_results").mkdir(exist_ok=True)
    combined.to_csv(csv_path)

    sep("═")
    print("\nFULL RESULTS\n")
    print(combined.to_string())
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
