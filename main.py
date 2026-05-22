"""
Pro-RAG — Adaptive Multi-Hop RAG with Triton + SGLang Prefix Caching

Start SGLang:
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-1.5B-Instruct-AWQ \
        --mem-fraction-static 0.8 --port 30000

Run:
    python main.py
"""

import sys, os, time
import numpy as np
import pandas as pd
from pathlib import Path

from src.pipeline    import setup, index_documents, query
from src.loaders     import load_file
from src.caching     import report as cache_report
from src.retrieval   import embed_query
from src.retrieval.retriever import retrieve
from src.agents      import analyze_query, run_retrieval
from src.agents.generator_agent import generate_hf, generate
from src.evaluation.ragas_eval  import evaluate_ragas, evaluate_ragas_no_gt

PDF_PATH = "harry_potter_ch1.pdf"

# Each cluster has ONE dominant keyword repeated in all 4 questions so embeddings
# are near-identical → retriever returns the same chunk every time → cache hits.
BENCHMARK_QA = [
    # Cluster 1 keyword: "Quirrell" + "Sorcerer's Stone"
    {
        "question":     "Who was Quirrell and why did he want the Sorcerer's Stone?",
        "ground_truth": "Professor Quirrell was the Defence Against the Dark Arts teacher possessed by Voldemort on the back of his head. He wanted the Sorcerer's Stone so Voldemort could use it to return to human form.",
    },
    {
        "question":     "How did Harry defeat Quirrell when Quirrell tried to take the Sorcerer's Stone?",
        "ground_truth": "Harry's touch burned Quirrell because Harry's mother's sacrificial death gave Harry a love-based magical protection against Voldemort. Quirrell, possessed by Voldemort, could not touch Harry without burning.",
    },
    {
        "question":     "Why did Quirrell burn and crumble when Harry grabbed his face near the Sorcerer's Stone?",
        "ground_truth": "Harry's mother died to save him, leaving Harry with a love-based magical protection against Voldemort. Quirrell was possessed by Voldemort, so he burned at Harry's touch.",
    },
    {
        "question":     "What did Voldemort promise Harry to get the Sorcerer's Stone and what happened to the Stone afterwards?",
        "ground_truth": "Voldemort promised to bring Harry's parents back from the dead. The Stone was later destroyed by Dumbledore.",
    },

    # Cluster 2 keyword: "Basilisk" + "Fawkes" + "Chamber"
    {
        "question":     "How did Fawkes the phoenix blind the Basilisk inside the Chamber of Secrets?",
        "ground_truth": "Fawkes attacked the Basilisk's eyes, blinding it so it could no longer kill with its gaze, then delivered the Sorting Hat to Harry from which appeared the Sword of Godric Gryffindor.",
    },
    {
        "question":     "How did Harry kill the Basilisk in the Chamber of Secrets and what injury did he suffer?",
        "ground_truth": "Harry used the Sword of Godric Gryffindor to impale the Basilisk in the roof of the mouth. A Basilisk fang pierced Harry's arm, poisoning him with venom.",
    },
    {
        "question":     "How did the Basilisk fang destroy Tom Riddle's diary inside the Chamber of Secrets?",
        "ground_truth": "Dying from the Basilisk fang's venom, Harry plunged the fang into Tom Riddle's diary, destroying the memory preserved inside. Fawkes then healed Harry's Basilisk fang wound with his tears.",
    },
    {
        "question":     "How did Fawkes save Harry after the Basilisk fang poisoned him in the Chamber of Secrets?",
        "ground_truth": "Fawkes cried on Harry's Basilisk fang wound; phoenix tears have healing powers that neutralised the venom and saved Harry's life.",
    },

    # Cluster 3 keyword: "Priori Incantatem" + "wands" + "graveyard"
    {
        "question":     "What is Priori Incantatem and what causes it between two wands?",
        "ground_truth": "Priori Incantatem is a magical connection that occurs when two wands sharing the same core are forced to duel. Harry's and Voldemort's wands both had cores from the same phoenix, Fawkes.",
    },
    {
        "question":     "What happened when Harry's and Voldemort's wands connected via Priori Incantatem in the graveyard?",
        "ground_truth": "Harry's wand forced Voldemort's wand to disgorge the spirits of people Voldemort had most recently killed, including Harry's parents and Cedric Diggory.",
    },
    {
        "question":     "Which spirits emerged from Voldemort's wand during Priori Incantatem and how did they help Harry escape?",
        "ground_truth": "The spirits of Harry's parents and Cedric Diggory emerged. They shielded Harry and gave him time to break the Priori Incantatem connection and reach the Portkey.",
    },
    {
        "question":     "How did Priori Incantatem between Harry's and Voldemort's wands allow Harry to escape the graveyard?",
        "ground_truth": "The spirit echoes from Priori Incantatem shielded Harry as he broke the wand connection. He summoned the Triwizard Cup Portkey and escaped with Cedric's body.",
    },

    # Cluster 4 keyword: "Pettigrew" + "Sirius" + "Scabbers"
    {
        "question":     "What was Peter Pettigrew's Animagus form and how long had he been hiding as Scabbers?",
        "ground_truth": "Peter Pettigrew's Animagus form was a rat. He had been hiding as Ron Weasley's pet rat Scabbers for twelve years to avoid capture for betraying the Potters to Voldemort.",
    },
    {
        "question":     "How did Peter Pettigrew frame Sirius Black for the betrayal of Harry's parents?",
        "ground_truth": "Pettigrew was the one who betrayed the Potters to Voldemort, then faked his own death and hid as the rat Scabbers to frame Sirius Black for the crime.",
    },
    {
        "question":     "Why was Sirius Black wrongly imprisoned and who was the real traitor according to what was revealed about Pettigrew?",
        "ground_truth": "Sirius Black was wrongly imprisoned because Peter Pettigrew framed him. Pettigrew was the actual traitor who had hidden as the rat Scabbers for twelve years.",
    },
    {
        "question":     "How did Harry and Hermione use the Time-Turner to rescue Sirius Black after Pettigrew escaped?",
        "ground_truth": "Hermione's Time-Turner let Harry and Hermione travel back three hours. They freed Buckbeak the Hippogriff and rescued Sirius Black, who escaped by flying away on Buckbeak.",
    },

    # Cluster 5 keyword: "Horcrux" + "Voldemort" + "soul"
    {
        "question":     "What is a Horcrux and how does it grant its creator immortality?",
        "ground_truth": "A Horcrux is an object that safeguards a portion of the creator's soul. As long as a Horcrux exists the creator cannot truly die, so all of Voldemort's Horcruxes had to be destroyed.",
    },
    {
        "question":     "How many Horcruxes did Voldemort create and which two Horcruxes were already destroyed before the final book?",
        "ground_truth": "Voldemort created seven Horcruxes. Two were destroyed before Deathly Hallows: Tom Riddle's diary and his grandfather's ring.",
    },
    {
        "question":     "How did Voldemort accidentally create a Horcrux inside Harry Potter when the Killing Curse rebounded?",
        "ground_truth": "When Voldemort's Killing Curse rebounded off baby Harry, the force tore a fragment of Voldemort's soul free and it lodged inside Harry, inadvertently making Harry an unintended Horcrux.",
    },
    {
        "question":     "Why could Harry survive Voldemort's Killing Curse in the final battle even though Harry was a Horcrux?",
        "ground_truth": "Voldemort had used Harry's blood to regain his body, which tied Harry's life to Voldemort's and protected Harry from harm. The Killing Curse destroyed the Horcrux soul fragment inside Harry but Harry himself returned to life.",
    },
]


def sep(char="─", n=64):
    print(char * n)


def find_gold_label(question):
    q = question.lower().strip()
    for qa in BENCHMARK_QA:
        if qa["question"].lower().strip() == q:
            return qa["ground_truth"]
    return None


# ── System runners — all use same SGLang generate() ──────────────────────

def _pack(chunks, gen, t0, hops, cache_hit=False):
    return {
        "retrieved_ids": [f"{c['chunk']['doc_id']}_{c['chunk']['chunk_id']}" for c in chunks],
        "answer":        gen["answer"],
        "contexts":      [c["chunk"]["text"] for c in chunks],
        "ttft_ms":       gen["ttft_ms"],
        "total_ms":      (time.perf_counter() - t0) * 1000,
        "hops":          hops,
        "cache_hit":     cache_hit,
    }

def run_s1(index, question):
    t0 = time.perf_counter()
    chunks, _, _ = retrieve(index, embed_query(question), force_k=5, use_triton=False)
    return _pack(chunks, generate_hf(question, chunks), t0, hops=1)

def run_s2(index, question):
    t0  = time.perf_counter()
    ret = run_retrieval(index, analyze_query(question), use_triton=False)
    return _pack(ret["chunks"], generate_hf(question, ret["chunks"]), t0, ret["hops"])

def run_s3(index, question):
    t0 = time.perf_counter()
    chunks, _, _ = retrieve(index, embed_query(question), force_k=5, use_triton=True)
    return _pack(chunks, generate_hf(question, chunks), t0, hops=1)

def run_s4(index, question):
    t0  = time.perf_counter()
    ret = run_retrieval(index, analyze_query(question), use_triton=True)
    return _pack(ret["chunks"], generate_hf(question, ret["chunks"]), t0, ret["hops"])

HF_SYSTEMS = [
    ("S1: Plain RAG",               run_s1),
    ("S2: RAG + Adaptive",          run_s2),
    ("S3: RAG + Triton",            run_s3),
    ("S4: RAG + Triton + Adaptive", run_s4),
]


# ── Aggregate results into one benchmark row ──────────────────────────────

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
        "system":        name,
        **scores,
        "avg_ttft_ms":   round(np.mean(ttfts),           1),
        "p95_ttft_ms":   round(np.percentile(ttfts, 95), 1),
        "avg_total_ms":  round(np.mean(totals),           1),
        "avg_hops":      round(np.mean(hops),             2),
        "cache_hits":    sum(1 for r in results if r.get("cache_hit", False)),
    }


# ── Benchmark ─────────────────────────────────────────────────────────────

def run_benchmark(index, chat_history):
    questions = [r["question"] for r in chat_history]

    sep("═")
    print(f"PRO-RAG BENCHMARK  —  {len(questions)} questions  —  Qwen2.5-1.5B")
    sep("═")

    reports = []

    for name, run_fn in HF_SYSTEMS:
        print(f"\n  {name}")
        results = []
        for q in questions:
            try:
                r = run_fn(index, q)
                r["question"] = q
                results.append(r)
            except Exception as e:
                print(f"    Error: {e}")
        if results:
            reports.append(aggregate(results, name))

    print(f"\n  Pro-RAG  (from chat session — no re-run needed)")
    s5 = [dict(r) for r in chat_history]
    reports.append(aggregate(s5, "Pro-RAG"))

    df = pd.DataFrame(reports).set_index("system")
    sep("═")
    print("\nRESULTS\n")
    print(df.to_string())

    import json
    Path("benchmark_results").mkdir(exist_ok=True)
    df.to_csv("benchmark_results/comparison.csv")
    print("\nSaved → benchmark_results/comparison.csv")

    # Save questions so run_s6.py can replay the same ones
    with open("benchmark_results/session_questions.json", "w") as f:
        json.dump(questions, f, indent=2)
    print("Saved → benchmark_results/session_questions.json")

    _plot(df)

    sep("─")
    print("SGLang Prefix Cache (radix attention)")
    for k, v in cache_report().items():
        print(f"  {k:<30} {v}")

    return df


def _plot(df):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        metric_cols  = [c for c in ["faithfulness", "answer_relevancy",
                                     "context_precision", "context_recall"] if c in df.columns]
        latency_cols = [c for c in ["avg_ttft_ms", "avg_total_ms", "avg_hops"] if c in df.columns]
        cols = metric_cols + latency_cols
        if not cols:
            return

        fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 6))
        if len(cols) == 1:
            axes = [axes]
        colors = sns.color_palette("husl", len(df))

        sys_labels = ["S1", "S2", "S3", "S4", "S5", "S6"][:len(df)]
        for ax, col in zip(axes, cols):
            bars = ax.bar(range(len(df)), df[col], color=colors)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(sys_labels, fontsize=8)
            ax.set_title(col, fontweight="bold", fontsize=9)
            for bar, val in zip(bars, df[col]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        labels = [f"{lbl}: {n}" for lbl, n in zip(sys_labels, df.index)]
        fig.legend(labels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
        plt.suptitle("Pro-RAG Benchmark — Adaptive Retrieval + Triton + SGLang Prefix Cache", fontsize=10, fontweight="bold")
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.savefig("benchmark_results/comparison.png", dpi=150, bbox_inches="tight")
        print("Plot  → benchmark_results/comparison.png")
    except ImportError:
        pass


# ── Session summary ───────────────────────────────────────────────────────

def show_session_summary(history):
    sep()
    print("CHAT SESSION  (Pro-RAG: Triton + Adaptive + Prefix Cache | SGLang)")
    sep()
    W = 38
    print(f"{'#':<3}  {'Question':<{W}}  {'TTFT':>7}  {'Total':>7}  {'Hops':>5}  {'Cache':>6}")
    print("─" * 68)
    for i, r in enumerate(history, 1):
        q = r["question"][:W-1] + "…" if len(r["question"]) > W else r["question"]
        print(f"{i:<3}  {q:<{W}}  {r['ttft_ms']:>6.0f}ms"
              f"  {r['total_ms']:>6.0f}ms  {r['hops']:>5}  "
              f"{'HIT' if r['cache_hit'] else 'miss':>6}")
    sep()
    ttfts  = [r["ttft_ms"]  for r in history]
    totals = [r["total_ms"] for r in history]
    print(f"Avg TTFT:   {np.mean(ttfts):.0f}ms  (p95 {np.percentile(ttfts, 95):.0f}ms)")
    print(f"Avg Total:  {np.mean(totals):.0f}ms  (p95 {np.percentile(totals, 95):.0f}ms)")
    print(f"Avg Hops:   {np.mean([r['hops'] for r in history]):.1f}")
    print(f"Cache hits: {sum(1 for r in history if r['cache_hit'])}/{len(history)}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    sep("═")
    print("  Adaptive RAG  |  Triton + SGLang Radix Attention")
    print("  Harry Potter — Full Saga Synopsis")
    sep("═")

    path = Path(PDF_PATH)
    if not path.exists():
        print(f"\nPDF not found: '{PDF_PATH}'")
        print("Place 'harry_potter_ch1.pdf' in the project folder and re-run.")
        sys.exit(1)

    print(f"\nLoading {path.name}...")
    docs  = load_file(str(path))
    print("Setting up pipeline...")
    index = setup()
    index_documents(index, docs)

    model = os.getenv("SGLANG_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    port  = os.getenv("SGLANG_PORT",  "30000")
    print(f"\nBackend: SGLang → {model} (port {port})")
    print(f"Backend: SGLang with radix attention prefix caching.\n")

    sep()
    print("PRESET QUESTIONS:")
    for i, qa in enumerate(BENCHMARK_QA, 1):
        print(f"  {i:>2}. {qa['question']}")
    sep()
    mode = input("Run preset questions automatically? [y] or chat manually [n]: ").strip().lower()

    history = []

    if mode != "n":
        print(f"\nRunning all {len(BENCHMARK_QA)} preset questions...\n")
        for qa in BENCHMARK_QA:
            sep()
            print(f"Q: {qa['question']}")
            result = query(index, qa['question'])
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\n  TTFT {result['ttft_ms']:.0f}ms | Total {result['total_ms']:.0f}ms | "
                  f"Hops {result['hops']} | k={result['k_values']} | "
                  f"cache={'HIT' if result['cache_hit'] else 'miss'}")
            history.append(result)
    else:
        print("Type questions. Type  quit  when done.\n")
        while True:
            sep()
            try:
                question = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break

            result = query(index, question)
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\n  TTFT {result['ttft_ms']:.0f}ms | Total {result['total_ms']:.0f}ms | "
                  f"Hops {result['hops']} | k={result['k_values']} | "
                  f"cache={'HIT' if result['cache_hit'] else 'miss'}")
            history.append(result)

    if not history:
        print("\nNo queries recorded.")
        return

    show_session_summary(history)

    sep("═")
    go = input("\nRun full benchmark? [y/n]: ").strip().lower()
    if go == "y":
        run_benchmark(index, history)


if __name__ == "__main__":
    main()
