"""
S6 Benchmark — run after main.py has completed S1-S5.

Requires:
  1. benchmark_results/session_questions.json   (saved by main.py)
  2. benchmark_results/comparison.csv           (saved by main.py)
  3. SGLang server running WITH speculative decoding:

     sglang serve \
      --model-path Qwen/Qwen2.5-1.5B-Instruct-AWQ \
      --quantization awq \
      --speculative-draft-model-path Qwen/Qwen2.5-0.5B-Instruct \
      --speculative-num-steps 5 \
      --mem-fraction-static 0.8 --port 30000

Run:
    python run_s6.py
"""

import sys, os, time, json
import numpy as np
import pandas as pd
from pathlib import Path

from src.pipeline    import setup, index_documents
from src.loaders     import load_file
from src.caching     import report as cache_report
from src.agents      import analyze_query, run_retrieval
from src.agents.generator_agent import generate
from src.evaluation.ragas_eval  import evaluate_ragas, evaluate_ragas_no_gt

PDF_PATH = "harry_potter_ch1.pdf"

BENCHMARK_QA = [
    # Cluster 1 keyword: "Quirrell" + "Sorcerer's Stone"
    {"question": "Who was Quirrell and why did he want the Sorcerer's Stone?",
     "ground_truth": "Professor Quirrell was the Defence Against the Dark Arts teacher possessed by Voldemort on the back of his head. He wanted the Sorcerer's Stone so Voldemort could use it to return to human form."},
    {"question": "How did Harry defeat Quirrell when Quirrell tried to take the Sorcerer's Stone?",
     "ground_truth": "Harry's touch burned Quirrell because Harry's mother's sacrificial death gave Harry a love-based magical protection against Voldemort. Quirrell, possessed by Voldemort, could not touch Harry without burning."},
    {"question": "Why did Quirrell burn and crumble when Harry grabbed his face near the Sorcerer's Stone?",
     "ground_truth": "Harry's mother died to save him, leaving Harry with a love-based magical protection against Voldemort. Quirrell was possessed by Voldemort, so he burned at Harry's touch."},
    {"question": "What did Voldemort promise Harry to get the Sorcerer's Stone and what happened to the Stone afterwards?",
     "ground_truth": "Voldemort promised to bring Harry's parents back from the dead. The Stone was later destroyed by Dumbledore."},
    # Cluster 2 keyword: "Basilisk" + "Fawkes" + "Chamber"
    {"question": "How did Fawkes the phoenix blind the Basilisk inside the Chamber of Secrets?",
     "ground_truth": "Fawkes attacked the Basilisk's eyes, blinding it so it could no longer kill with its gaze, then delivered the Sorting Hat to Harry from which appeared the Sword of Godric Gryffindor."},
    {"question": "How did Harry kill the Basilisk in the Chamber of Secrets and what injury did he suffer?",
     "ground_truth": "Harry used the Sword of Godric Gryffindor to impale the Basilisk in the roof of the mouth. A Basilisk fang pierced Harry's arm, poisoning him with venom."},
    {"question": "How did the Basilisk fang destroy Tom Riddle's diary inside the Chamber of Secrets?",
     "ground_truth": "Dying from the Basilisk fang's venom, Harry plunged the fang into Tom Riddle's diary, destroying the memory preserved inside. Fawkes then healed Harry's Basilisk fang wound with his tears."},
    {"question": "How did Fawkes save Harry after the Basilisk fang poisoned him in the Chamber of Secrets?",
     "ground_truth": "Fawkes cried on Harry's Basilisk fang wound; phoenix tears have healing powers that neutralised the venom and saved Harry's life."},
    # Cluster 3 keyword: "Priori Incantatem" + "wands" + "graveyard"
    {"question": "What is Priori Incantatem and what causes it between two wands?",
     "ground_truth": "Priori Incantatem is a magical connection that occurs when two wands sharing the same core are forced to duel. Harry's and Voldemort's wands both had cores from the same phoenix, Fawkes."},
    {"question": "What happened when Harry's and Voldemort's wands connected via Priori Incantatem in the graveyard?",
     "ground_truth": "Harry's wand forced Voldemort's wand to disgorge the spirits of people Voldemort had most recently killed, including Harry's parents and Cedric Diggory."},
    {"question": "Which spirits emerged from Voldemort's wand during Priori Incantatem and how did they help Harry escape?",
     "ground_truth": "The spirits of Harry's parents and Cedric Diggory emerged. They shielded Harry and gave him time to break the Priori Incantatem connection and reach the Portkey."},
    {"question": "How did Priori Incantatem between Harry's and Voldemort's wands allow Harry to escape the graveyard?",
     "ground_truth": "The spirit echoes from Priori Incantatem shielded Harry as he broke the wand connection. He summoned the Triwizard Cup Portkey and escaped with Cedric's body."},
    # Cluster 4 keyword: "Pettigrew" + "Sirius" + "Scabbers"
    {"question": "What was Peter Pettigrew's Animagus form and how long had he been hiding as Scabbers?",
     "ground_truth": "Peter Pettigrew's Animagus form was a rat. He had been hiding as Ron Weasley's pet rat Scabbers for twelve years to avoid capture for betraying the Potters to Voldemort."},
    {"question": "How did Peter Pettigrew frame Sirius Black for the betrayal of Harry's parents?",
     "ground_truth": "Pettigrew was the one who betrayed the Potters to Voldemort, then faked his own death and hid as the rat Scabbers to frame Sirius Black for the crime."},
    {"question": "Why was Sirius Black wrongly imprisoned and who was the real traitor according to what was revealed about Pettigrew?",
     "ground_truth": "Sirius Black was wrongly imprisoned because Peter Pettigrew framed him. Pettigrew was the actual traitor who had hidden as the rat Scabbers for twelve years."},
    {"question": "How did Harry and Hermione use the Time-Turner to rescue Sirius Black after Pettigrew escaped?",
     "ground_truth": "Hermione's Time-Turner let Harry and Hermione travel back three hours. They freed Buckbeak the Hippogriff and rescued Sirius Black, who escaped by flying away on Buckbeak."},
    # Cluster 5 keyword: "Horcrux" + "Voldemort" + "soul"
    {"question": "What is a Horcrux and how does it grant its creator immortality?",
     "ground_truth": "A Horcrux is an object that safeguards a portion of the creator's soul. As long as a Horcrux exists the creator cannot truly die, so all of Voldemort's Horcruxes had to be destroyed."},
    {"question": "How many Horcruxes did Voldemort create and which two Horcruxes were already destroyed before the final book?",
     "ground_truth": "Voldemort created seven Horcruxes. Two were destroyed before Deathly Hallows: Tom Riddle's diary and his grandfather's ring."},
    {"question": "How did Voldemort accidentally create a Horcrux inside Harry Potter when the Killing Curse rebounded?",
     "ground_truth": "When Voldemort's Killing Curse rebounded off baby Harry, the force tore a fragment of Voldemort's soul free and it lodged inside Harry, inadvertently making Harry an unintended Horcrux."},
    {"question": "Why could Harry survive Voldemort's Killing Curse in the final battle even though Harry was a Horcrux?",
     "ground_truth": "Voldemort had used Harry's blood to regain his body, which tied Harry's life to Voldemort's and protected Harry from harm. The Killing Curse destroyed the Horcrux soul fragment inside Harry but Harry himself returned to life."},
]


def sep(char="─", n=64):
    print(char * n)


def find_gold_label(question):
    q = question.lower().strip()
    for qa in BENCHMARK_QA:
        if qa["question"].lower().strip() == q:
            return qa["ground_truth"]
    return None


def _pack(chunks, gen, t0, hops):
    return {
        "retrieved_ids": [f"{c['chunk']['doc_id']}_{c['chunk']['chunk_id']}" for c in chunks],
        "answer":        gen["answer"],
        "contexts":      [c["chunk"]["text"] for c in chunks],
        "ttft_ms":       gen["ttft_ms"],
        "total_ms":      (time.perf_counter() - t0) * 1000,
        "hops":          hops,
    }


def run_s6(index, question):
    t0  = time.perf_counter()
    ret = run_retrieval(index, analyze_query(question), use_triton=True)
    gen = generate(question, ret["chunks"])
    return _pack(ret["chunks"], gen, t0, ret["hops"])


def aggregate_s6(results):
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
        "system":       "S6: RAG + Triton + Adaptive + Prefix Cache + SpecDec",
        **scores,
        "avg_ttft_ms":  round(np.mean(ttfts),           1),
        "p95_ttft_ms":  round(np.percentile(ttfts, 95), 1),
        "avg_total_ms": round(np.mean(totals),           1),
        "avg_hops":     round(np.mean(hops),             2),
        "cache_hits":   0,
    }


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

        sys_labels = ["S1", "S2", "S3", "S4", "S5", "S6"][:len(df)]
        fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 6))
        if len(cols) == 1:
            axes = [axes]
        colors = sns.color_palette("husl", len(df))

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
        plt.suptitle(
            "RAG Benchmark — HP Saga | S1-S4: HF API  S5: SGLang+Cache  S6: +SpecDec",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.savefig("benchmark_results/comparison.png", dpi=150, bbox_inches="tight")
        print("Plot  → benchmark_results/comparison.png  (all 6 systems)")
    except ImportError:
        pass


def main():
    questions_path = Path("benchmark_results/session_questions.json")
    csv_path       = Path("benchmark_results/comparison.csv")

    if not questions_path.exists():
        print("benchmark_results/session_questions.json not found.")
        print("Run main.py first and complete the S1-S5 benchmark.")
        sys.exit(1)

    if not csv_path.exists():
        print("benchmark_results/comparison.csv not found.")
        print("Run main.py first and complete the S1-S5 benchmark.")
        sys.exit(1)

    with open(questions_path) as f:
        questions = json.load(f)

    sep("═")
    print(f"S6 BENCHMARK  —  {len(questions)} questions  —  SGLang + Prefix Cache + Speculative Decoding")
    sep("═")
    print("Make sure SGLang is running WITH speculative decoding:")
    print("  sglang serve \\")
    print("      --model-path Qwen/Qwen2.5-1.5B-Instruct-AWQ \\")
    print("      --quantization awq \\")
    print("      --speculative-draft-model-path Qwen/Qwen2.5-0.5B-Instruct \\")
    print("      --speculative-num-steps 5 \\")
    print("      --mem-fraction-static 0.8 --port 30000")
    sep()

    path = Path(PDF_PATH)
    if not path.exists():
        print(f"PDF not found: '{PDF_PATH}' — place it in the project root.")
        sys.exit(1)

    print(f"\nLoading {path.name}...")
    docs  = load_file(str(path))
    print("Setting up pipeline...")
    index = setup()
    index_documents(index, docs)
    print()

    results = []
    for q in questions:
        try:
            r = run_s6(index, q)
            r["question"] = q
            results.append(r)
            print(f"  ✓ {q[:60]}{'…' if len(q) > 60 else ''}")
            print(f"    TTFT {r['ttft_ms']:.0f}ms | Total {r['total_ms']:.0f}ms | Hops {r['hops']}")
        except Exception as e:
            print(f"  ✗ Error on '{q[:50]}': {e}")

    if not results:
        print("\nNo results collected — is the SGLang server running?")
        sys.exit(1)

    print()
    s6_row = aggregate_s6(results)

    # Merge with existing S1-S5 results
    existing = pd.read_csv(csv_path, index_col="system")
    s6_df    = pd.DataFrame([s6_row]).set_index("system")

    # Drop old S6 row if re-running
    existing = existing[~existing.index.str.startswith("S6")]

    combined = pd.concat([existing, s6_df])
    combined.to_csv(csv_path)

    sep("═")
    print("\nFULL RESULTS (S1 → S6)\n")
    print(combined.to_string())
    print(f"\nUpdated → {csv_path}")

    _plot(combined)

    sep("─")
    print("SGLang Prefix Cache + Speculative Decoding report:")
    for k, v in cache_report().items():
        print(f"  {k:<30} {v}")


if __name__ == "__main__":
    main()
