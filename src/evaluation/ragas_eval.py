"""
RAGAS evaluation — uses local SGLang server as the judge LLM.
Same model as generation so everything stays self-contained.
"""

import os
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()


def _build_evaluator():
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    port  = os.getenv("SGLANG_PORT",  "30000")
    model = os.getenv("SGLANG_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    llm = LangchainLLMWrapper(
        ChatOpenAI(model=model, base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
    )
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )
    return llm, emb


def evaluate_ragas_no_gt(samples):
    """Faithfulness + answer_relevancy — no ground truth needed."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy

    llm, emb = _build_evaluator()
    faithfulness.llm        = llm
    answer_relevancy.llm    = llm
    answer_relevancy.embeddings = emb

    dataset = Dataset.from_dict({
        "question": [s["question"] for s in samples],
        "answer":   [s["answer"]   for s in samples],
        "contexts": [s["contexts"] for s in samples],
    })
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    return {
        "faithfulness":     round(result["faithfulness"],     4),
        "answer_relevancy": round(result["answer_relevancy"], 4),
    }


def evaluate_ragas(samples):
    """Full 4-metric RAGAS — requires ground_truth in each sample."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    llm, emb = _build_evaluator()
    faithfulness.llm        = llm
    answer_relevancy.llm    = llm
    answer_relevancy.embeddings = emb
    context_precision.llm   = llm
    context_recall.llm      = llm

    dataset = Dataset.from_dict({
        "question":     [s["question"]     for s in samples],
        "answer":       [s["answer"]       for s in samples],
        "contexts":     [s["contexts"]     for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    })
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return {
        "faithfulness":      round(result["faithfulness"],      4),
        "answer_relevancy":  round(result["answer_relevancy"],  4),
        "context_precision": round(result["context_precision"], 4),
        "context_recall":    round(result["context_recall"],    4),
    }
