"""
Pro-RAG full benchmark — runs all 50 questions, captures every metric.

Requires SGLang running on port 30000:
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-1.5B-Instruct-AWQ \
        --mem-fraction-static 0.8 --port 30000

Run:
    python run_prorag.py
"""

import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path

from src.pipeline  import setup, index_documents, query
from src.loaders   import load_file
from src.caching   import report as cache_report
from src.evaluation.ragas_eval import evaluate_ragas

PDF_PATH = "harry_potter_ch1.pdf"

BENCHMARK_QA = [
    # ── Cluster 1: Quirrell ──────────────────────────────────────────────
    {"question": "Who was Quirrell and what secret was he hiding?",
     "ground_truth": "Professor Quirrell was the Defence Against the Dark Arts teacher at Hogwarts who was secretly sharing his body with Voldemort, who had attached himself to the back of Quirrell's head beneath his turban."},
    {"question": "Why was Quirrell secretly working for Voldemort?",
     "ground_truth": "Quirrell had sought out Voldemort in the forests of Albania and was seduced by the promise of power. Voldemort parasitically took over Quirrell's body, and Quirrell served him in exchange for what he believed would be a share of that power."},
    {"question": "What was Quirrell's true motive for wanting the Sorcerer's Stone?",
     "ground_truth": "Quirrell wanted the Sorcerer's Stone on behalf of Voldemort so that Voldemort could use the Elixir of Life produced by the Stone to regain a physical body and return to full power."},
    {"question": "How did Harry overpower Quirrell in the underground chamber?",
     "ground_truth": "Harry grabbed Quirrell's face and hands with his bare hands. His touch burned Quirrell because Lily Potter's sacrificial love protected Harry, making him deadly to Quirrell who was possessed by Voldemort."},
    {"question": "What happened to Quirrell after Harry grabbed his face?",
     "ground_truth": "Quirrell's body crumbled and turned to dust when Harry grabbed him, because the love protection in Harry's skin was lethal to someone sharing a body with Voldemort. Quirrell died and Voldemort's spirit fled."},

    # ── Cluster 2: Sorcerer's Stone ──────────────────────────────────────
    {"question": "What is the Sorcerer's Stone?",
     "ground_truth": "The Sorcerer's Stone (Philosopher's Stone) is a legendary alchemical object created by Nicolas Flamel that produces the Elixir of Life, which grants immortality, and can also transform any metal into pure gold."},
    {"question": "What power does the Sorcerer's Stone grant its owner?",
     "ground_truth": "The Sorcerer's Stone produces the Elixir of Life, which grants the drinker immortality and indefinite life extension. It can also transmute any metal into pure gold."},
    {"question": "Why did Voldemort want the Sorcerer's Stone?",
     "ground_truth": "Voldemort needed the Sorcerer's Stone to obtain the Elixir of Life so he could create a permanent body and achieve true immortality, restoring him from his weakened wraith-like state after his curse rebounded off Harry."},
    {"question": "What did Dumbledore do with the Sorcerer's Stone after Harry defeated Quirrell?",
     "ground_truth": "Dumbledore arranged for the Sorcerer's Stone to be destroyed. Nicolas Flamel and his wife Perenelle, who depended on the Elixir of Life, had enough stock to set their affairs in order before dying naturally."},
    {"question": "What did Voldemort promise Harry in exchange for the Sorcerer's Stone?",
     "ground_truth": "Voldemort offered to bring Harry's parents back from the dead if Harry handed over the Sorcerer's Stone. Harry refused."},

    # ── Cluster 3: Basilisk ──────────────────────────────────────────────
    {"question": "What is the Basilisk in the Chamber of Secrets?",
     "ground_truth": "The Basilisk is a giant serpent, the King of Serpents, which Tom Riddle had placed inside the Chamber of Secrets at Hogwarts. It was used to attack Muggle-born students by looking at them with its deadly gaze."},
    {"question": "How did the Basilisk attack students at Hogwarts?",
     "ground_truth": "The Basilisk's direct gaze kills instantly, but students survived because they only saw its eyes indirectly — through a mirror, through a ghost, in a camera reflection, or through water — which merely petrified them."},
    {"question": "How did Harry kill the Basilisk in the Chamber of Secrets?",
     "ground_truth": "Harry pulled the Sword of Godric Gryffindor from the Sorting Hat delivered by Fawkes, then drove the sword through the roof of the Basilisk's mouth, killing it, though a fang pierced Harry's arm in the process."},
    {"question": "What injury did Harry suffer when he killed the Basilisk?",
     "ground_truth": "A Basilisk fang pierced Harry's arm during the fight. Basilisk venom is lethal and began killing Harry until Fawkes cried on the wound, as phoenix tears can heal any injury including Basilisk venom poisoning."},
    {"question": "How did Fawkes help Harry fight the Basilisk in the Chamber of Secrets?",
     "ground_truth": "Fawkes blinded the Basilisk by attacking its eyes, removing its lethal gaze. He then delivered the Sorting Hat to Harry (from which Harry drew the Sword of Gryffindor), and later cried on Harry's fang wound to heal the Basilisk venom."},

    # ── Cluster 4: Fawkes ────────────────────────────────────────────────
    {"question": "Who is Fawkes and what role did Fawkes play in the Chamber of Secrets?",
     "ground_truth": "Fawkes is Dumbledore's phoenix. In the Chamber of Secrets he blinded the Basilisk, delivered the Sorting Hat (containing the Sword of Gryffindor) to Harry, healed Harry's Basilisk fang wound with his tears, and carried Harry, Ron, Ginny, and Lockhart out of the Chamber."},
    {"question": "How did Fawkes heal Harry's Basilisk fang wound in the Chamber of Secrets?",
     "ground_truth": "Fawkes cried on the Basilisk fang wound in Harry's arm. Phoenix tears have healing properties powerful enough to neutralize Basilisk venom, which saved Harry's life."},
    {"question": "What did Fawkes deliver to Harry inside the Chamber of Secrets?",
     "ground_truth": "Fawkes delivered the old Sorting Hat to Harry. Harry then pulled the Sword of Godric Gryffindor out of the hat, which he used to kill the Basilisk."},
    {"question": "Why were Fawkes's tears able to save Harry from the Basilisk venom?",
     "ground_truth": "Phoenix tears are a powerful healing agent in the wizarding world. They can cure any wound, including the effects of Basilisk venom, which is otherwise fatal. Fawkes cried on Harry's fang wound and neutralized the venom."},
    {"question": "How did Fawkes blind the Basilisk during Harry's fight in the Chamber of Secrets?",
     "ground_truth": "Fawkes flew at the Basilisk and used his talons to attack and gouge out the Basilisk's eyes, blinding it. This removed the Basilisk's lethal gaze so Harry could face the serpent without being killed."},

    # ── Cluster 5: Pettigrew ─────────────────────────────────────────────
    {"question": "Who was Peter Pettigrew hiding as and for how long?",
     "ground_truth": "Peter Pettigrew had been hiding as Scabbers, Ron Weasley's pet rat, for twelve years, ever since he faked his death to frame Sirius Black and escape punishment for betraying the Potters."},
    {"question": "What was Peter Pettigrew's Animagus animal form?",
     "ground_truth": "Peter Pettigrew's Animagus form was a rat. He had registered — or rather unregistered — this ability, having learned to transform alongside James Potter, Sirius Black, and Remus Lupin at Hogwarts."},
    {"question": "What crime did Peter Pettigrew commit that was blamed on Sirius Black?",
     "ground_truth": "Peter Pettigrew betrayed the Potters' hiding place to Voldemort. After Voldemort's fall, he faked his own death, cut off his own finger as evidence, transformed into his rat form, and framed Sirius Black for his murder and the betrayal."},
    {"question": "Why did Peter Pettigrew hide as the rat Scabbers for twelve years?",
     "ground_truth": "Pettigrew hid as Scabbers to avoid discovery and punishment for betraying the Potters. He feared Voldemort's remaining followers and Sirius Black, and needed to stay hidden until Voldemort could rise again."},
    {"question": "How was Peter Pettigrew exposed as the real traitor who had been hiding as Scabbers?",
     "ground_truth": "Remus Lupin and Sirius Black cornered Pettigrew in the Shrieking Shack. Lupin recognized him from the Marauder's Map. Hermione forced him to transform from rat to human, and his own confession plus physical evidence (missing finger) exposed him."},

    # ── Cluster 6: Sirius Black ──────────────────────────────────────────
    {"question": "Who is Sirius Black and what is his relationship to Harry Potter?",
     "ground_truth": "Sirius Black is Harry Potter's godfather, a pure-blood wizard and best friend of Harry's father James. He was wrongly imprisoned in Azkaban for twelve years for crimes he did not commit."},
    {"question": "Why was Sirius Black sent to Azkaban prison?",
     "ground_truth": "Sirius was framed by Peter Pettigrew for betraying the Potters to Voldemort and for the murder of twelve Muggles. Pettigrew staged his own death and let Sirius take the blame without a trial."},
    {"question": "How was Sirius Black proved innocent of betraying the Potters?",
     "ground_truth": "Sirius was never officially cleared in the legal system during the books — Peter Pettigrew escaped before formal testimony could be given. His innocence was known to Harry, Hermione, Ron, Lupin, and Dumbledore, but he died a fugitive."},
    {"question": "How did Sirius Black escape after Pettigrew fled and the Dementors attacked?",
     "ground_truth": "When Dementors overwhelmed Sirius, Harry and Hermione used the Time-Turner to travel back three hours. Harry then cast a powerful Patronus Charm (a stag) to drive off the Dementors, and they freed Sirius who escaped on Buckbeak the Hippogriff."},
    {"question": "What is Sirius Black's connection to the flying motorcycle mentioned in the series?",
     "ground_truth": "The flying motorcycle belonged to Sirius Black. He lent it to Hagrid on the night Voldemort fell so Hagrid could transport baby Harry from Godric's Hollow to the Dursleys. It was later inherited by Harry."},

    # ── Cluster 7: Priori Incantatem ─────────────────────────────────────
    {"question": "What is Priori Incantatem?",
     "ground_truth": "Priori Incantatem is the rare magical effect that occurs when two wands sharing the same core are forced to duel each other. The wands lock in a golden thread of magic and the weaker wand is forced to echo the last spells cast by the stronger."},
    {"question": "What causes Priori Incantatem to happen between two wands?",
     "ground_truth": "Priori Incantatem occurs when two wands with the same magical core — twin cores — are made to fight each other. Harry's and Voldemort's wands both contain a feather from the same phoenix, Fawkes, triggering the connection."},
    {"question": "What happened during Priori Incantatem between Harry's and Voldemort's wands in the graveyard?",
     "ground_truth": "A golden thread of light connected the wands and formed a dome of light. Voldemort's wand was forced to regurgitate echoes of its last victims as ghost-like shadows, including Cedric Diggory, Frank Bryce, Bertha Jorkins, and Harry's parents."},
    {"question": "Which spirits appeared from Voldemort's wand during Priori Incantatem?",
     "ground_truth": "The shadows of Cedric Diggory, Bertha Jorkins, Frank Bryce, and Harry's parents — Lily and James Potter — emerged from Voldemort's wand as echo spirits in reverse order of when they were killed."},
    {"question": "How did Priori Incantatem help Harry escape from Voldemort in the graveyard?",
     "ground_truth": "The spirit echoes from Priori Incantatem, including Harry's parents, shielded Harry and urged him to break the connection. When Harry did, the echoes bought him time to reach the Triwizard Cup Portkey and escape with Cedric's body."},

    # ── Cluster 8: Time-Turner ───────────────────────────────────────────
    {"question": "What is a Time-Turner and who used one in Prisoner of Azkaban?",
     "ground_truth": "A Time-Turner is a magical hourglass device that allows the user to travel back in time. Hermione Granger used one throughout her third year at Hogwarts to attend multiple classes simultaneously, with Ministry of Magic permission."},
    {"question": "How did Hermione use the Time-Turner throughout Prisoner of Azkaban?",
     "ground_truth": "Hermione wore the Time-Turner around her neck and turned it to go back in time, allowing her to attend extra classes that ran at the same hour. Each turn of the hourglass moved the user back one hour."},
    {"question": "What did Harry and Hermione achieve by using the Time-Turner to travel back three hours?",
     "ground_truth": "Harry and Hermione traveled back three hours and saved Buckbeak the Hippogriff from execution by untying him from his post. They then used Buckbeak to rescue Sirius Black from the tower where he was imprisoned awaiting the Dementor's Kiss."},
    {"question": "How did Harry realise he was the one who cast the stag Patronus after using the Time-Turner?",
     "ground_truth": "When Harry witnessed the Dementor attack across the lake from the past timeline, he waited for someone to cast a Patronus. When no one came, he realized he had seen himself do it in the future, stepped forward, and cast the full corporeal stag Patronus."},
    {"question": "Why did Harry and Hermione need the Time-Turner to save both Sirius and Buckbeak?",
     "ground_truth": "Buckbeak was scheduled for execution and Sirius was imprisoned to receive the Dementor's Kiss, both seemingly irreversible in the present timeline. Traveling back in time was the only way to save both without the authorities knowing, as Dumbledore hinted."},

    # ── Cluster 9: Horcrux ───────────────────────────────────────────────
    {"question": "What is a Horcrux?",
     "ground_truth": "A Horcrux is a dark magical object in which a wizard conceals a fragment of their soul by committing murder, which tears the soul. As long as the Horcrux survives the wizard cannot die, making it the darkest form of dark magic."},
    {"question": "How does a Horcrux grant its creator immortality?",
     "ground_truth": "By splitting their soul and hiding a piece in a Horcrux, the creator cannot be truly killed as long as the Horcrux survives. Even if their body is destroyed, the soul fragment anchors them to the living world."},
    {"question": "How many Horcruxes did Voldemort create?",
     "ground_truth": "Voldemort intended to create six Horcruxes to split his soul into seven parts (including the piece that remained in his body). He accidentally created a seventh when his Killing Curse rebounded off Harry, making Harry an unintentional Horcrux."},
    {"question": "Which of Voldemort's Horcruxes were destroyed before Deathly Hallows?",
     "ground_truth": "Tom Riddle's diary was destroyed by Harry with a Basilisk fang in Chamber of Secrets. Marvolo Gaunt's ring was destroyed by Dumbledore using the Sword of Gryffindor. These two were gone before the events of Deathly Hallows."},
    {"question": "How did Harry accidentally become one of Voldemort's Horcruxes?",
     "ground_truth": "When Voldemort's Killing Curse rebounded off baby Harry, the force tore a fragment of Voldemort's unstable soul free. That fragment latched onto the only living being nearby — Harry — making him an unintentional Horcrux."},

    # ── Cluster 10: Deathly Hallows ──────────────────────────────────────
    {"question": "What are the Deathly Hallows?",
     "ground_truth": "The Deathly Hallows are three legendary magical objects created by Death (or the Peverell brothers): the Elder Wand (the most powerful wand), the Resurrection Stone (which summons the dead), and the Invisibility Cloak (which hides the wearer from Death)."},
    {"question": "What are the three objects that make up the Deathly Hallows?",
     "ground_truth": "The three Deathly Hallows are the Elder Wand, the most powerful wand ever made; the Resurrection Stone, which can summon the shadows of the dead; and the Cloak of Invisibility, which renders the wearer completely invisible even to Death."},
    {"question": "What does the Elder Wand do and why did Voldemort seek it among the Deathly Hallows?",
     "ground_truth": "The Elder Wand is the most powerful wand in existence, said to be unbeatable in a duel. Voldemort sought it because he believed it would allow him to cast a Killing Curse that would finally kill Harry Potter."},
    {"question": "What does the Resurrection Stone do in the Deathly Hallows?",
     "ground_truth": "The Resurrection Stone can summon the shadows or echoes of dead loved ones back to the living world. They are not truly alive but can speak to the living. Harry used it in the Forbidden Forest to speak with his parents, Sirius, and Lupin before facing Voldemort."},
    {"question": "Which of the Deathly Hallows was Voldemort pursuing and why?",
     "ground_truth": "Voldemort pursued the Elder Wand, believing it would give him the power to kill Harry once and for all. He broke into Dumbledore's tomb at Hogwarts to steal it, not knowing the wand's true allegiance had already passed to Draco Malfoy and then to Harry."},
]


def sep(char="─", n=64):
    print(char * n)


def main():
    sep("═")
    print("  Pro-RAG Benchmark  |  50 questions  |  Full metrics")
    sep("═")

    path = Path(PDF_PATH)
    if not path.exists():
        print(f"\nPDF not found: '{PDF_PATH}'")
        sys.exit(1)

    print(f"\nLoading {path.name}...")
    docs  = load_file(str(path))
    print("Setting up pipeline...")
    index = setup()
    index_documents(index, docs)
    print()

    results = []
    sep()
    print(f"Running {len(BENCHMARK_QA)} questions through Pro-RAG...\n")
    sep()

    for i, qa in enumerate(BENCHMARK_QA, 1):
        q = qa["question"]
        try:
            r = query(index, q)
            r["ground_truth"] = qa["ground_truth"]
            results.append(r)
            cache_str = "HIT" if r["cache_hit"] else "miss"
            print(f"  [{i:>2}/50] TTFT {r['ttft_ms']:>6.0f}ms | Total {r['total_ms']:>7.0f}ms | "
                  f"Hops {r['hops']} | cache={cache_str}")
            print(f"         {q[:80]}")
        except Exception as e:
            print(f"  [{i:>2}/50] ERROR: {e}")
            print(f"         {q[:80]}")

    if not results:
        print("\nNo results — is SGLang running on port 30000?")
        sys.exit(1)

    # ── Session summary ──────────────────────────────────────────────────
    ttfts   = [r["ttft_ms"]  for r in results]
    totals  = [r["total_ms"] for r in results]
    hops    = [r["hops"]     for r in results]
    hits    = sum(1 for r in results if r["cache_hit"])

    sep("═")
    print("\nSESSION SUMMARY\n")
    print(f"  Questions completed : {len(results)}/50")
    print(f"  Avg TTFT            : {np.mean(ttfts):.0f} ms  (p95 {np.percentile(ttfts, 95):.0f} ms)")
    print(f"  Avg Total Latency   : {np.mean(totals):.0f} ms  (p95 {np.percentile(totals, 95):.0f} ms)")
    print(f"  Avg Hops            : {np.mean(hops):.1f}")
    print(f"  Prefix Cache Hits   : {hits}/{len(results)}")

    # ── Per-cluster cache hit breakdown ──────────────────────────────────
    clusters = [
        ("Quirrell",          results[0:5]),
        ("Sorcerer's Stone",  results[5:10]),
        ("Basilisk",          results[10:15]),
        ("Fawkes",            results[15:20]),
        ("Pettigrew",         results[20:25]),
        ("Sirius Black",      results[25:30]),
        ("Priori Incantatem", results[30:35]),
        ("Time-Turner",       results[35:40]),
        ("Horcrux",           results[40:45]),
        ("Deathly Hallows",   results[45:50]),
    ]
    sep()
    print("\nCACHE HITS BY CLUSTER\n")
    for name, grp in clusters:
        h = sum(1 for r in grp if r["cache_hit"])
        print(f"  {name:<20} {h}/{len(grp)}")

    # ── Cache speed stats ─────────────────────────────────────────────────
    sep()
    print("\nPREFIX CACHE STATS\n")
    for k, v in cache_report().items():
        print(f"  {k:<30} {v}")

    # ── RAGAS ─────────────────────────────────────────────────────────────
    sep()
    print("\nRunning RAGAS evaluation (4 metrics, using SGLang as judge)...\n")
    ragas_samples = [
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in results
    ]
    try:
        ragas_scores = evaluate_ragas(ragas_samples)
        print("\nRAGAS SCORES\n")
        for metric, score in ragas_scores.items():
            print(f"  {metric:<25} {score:.4f}")
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        ragas_scores = {}

    # ── Save results ──────────────────────────────────────────────────────
    Path("benchmark_results").mkdir(exist_ok=True)

    # Full per-question CSV
    rows = []
    for i, r in enumerate(results, 1):
        rows.append({
            "q_num":        i,
            "question":     r["question"],
            "answer":       r["answer"],
            "ground_truth": r["ground_truth"],
            "ttft_ms":      round(r["ttft_ms"],   1),
            "total_ms":     round(r["total_ms"],  1),
            "hops":         r["hops"],
            "cache_hit":    r["cache_hit"],
            "complexity":   round(r.get("complexity", 0), 3),
            "confidence":   round(r.get("confidence",  0), 3),
            "num_chunks":   r.get("num_chunks", 0),
        })
    detail_df = pd.DataFrame(rows)
    detail_df.to_csv("benchmark_results/prorag_results.csv", index=False)
    print(f"\nDetailed results → benchmark_results/prorag_results.csv")

    # Summary row
    summary = {
        "system":           "Pro-RAG",
        **ragas_scores,
        "avg_ttft_ms":      round(np.mean(ttfts),            1),
        "p95_ttft_ms":      round(np.percentile(ttfts, 95),  1),
        "avg_total_ms":     round(np.mean(totals),           1),
        "p95_total_ms":     round(np.percentile(totals, 95), 1),
        "avg_hops":         round(np.mean(hops),             2),
        "cache_hits":       hits,
        "cache_hit_rate":   round(hits / len(results), 3),
        "questions":        len(results),
    }
    pd.DataFrame([summary]).to_csv("benchmark_results/summary.csv", index=False)
    print(f"Summary          → benchmark_results/summary.csv")

    sep("═")
    print("\nFINAL RESULTS\n")
    for k, v in summary.items():
        print(f"  {k:<25} {v}")
    sep("═")


if __name__ == "__main__":
    main()
