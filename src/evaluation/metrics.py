import math
import numpy as np


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids[:k] if r in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids, relevant_ids):
    for rank, r in enumerate(retrieved_ids, 1):
        if r in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    def dcg(ids):
        return sum(1.0 / math.log2(i + 2) for i, r in enumerate(ids[:k]) if r in relevant_ids)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg(retrieved_ids) / ideal if ideal > 0 else 0.0


def context_precision(retrieved_ids, relevant_ids, k):
    hits = sum(1 for r in retrieved_ids[:k] if r in relevant_ids)
    return hits / min(k, len(retrieved_ids)) if retrieved_ids else 0.0


def faithfulness(answer, chunk_texts):
    context_words = set()
    for t in chunk_texts:
        context_words |= set(t.lower().split())
    answer_words = set(answer.lower().split())
    return len(answer_words & context_words) / len(answer_words) if answer_words else 0.0


def answer_relevancy(answer, query):
    q_words = set(query.lower().split())
    a_words = set(answer.lower().split())
    return len(q_words & a_words) / len(q_words) if q_words else 0.0


def all_retrieval_metrics(retrieved_ids, relevant_ids):
    return {
        "recall@1":  recall_at_k(retrieved_ids, relevant_ids, 1),
        "recall@3":  recall_at_k(retrieved_ids, relevant_ids, 3),
        "recall@5":  recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall@10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "mrr":       mrr(retrieved_ids, relevant_ids),
        "ndcg@5":    ndcg_at_k(retrieved_ids, relevant_ids, 5),
        "ndcg@10":   ndcg_at_k(retrieved_ids, relevant_ids, 10),
        "ctx_prec@5": context_precision(retrieved_ids, relevant_ids, 5),
    }
