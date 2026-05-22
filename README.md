# Pro-RAG

Production-grade Retrieval-Augmented Generation combining adaptive multi-hop retrieval, custom Triton CUDA kernels, and prefix-cached LLM inference. Built to answer questions over PDF documents with low latency and measurable accuracy.

---

## Architecture

![Pro-RAG Architecture](diagram.png)

---

## Results

| Metric | Value |
|---|---|
| Avg Time to First Token | 909 ms |
| p95 TTFT | 1,424 ms |
| Avg Total Latency | 4,118 ms |
| Prefix Cache Hit Rate | 38% (19 / 50 queries) |
| Avg Retrieval Hops | 1.9 |

Benchmarked on a 50-question clustered set (10 topic clusters x 5 questions) against a local Qwen2.5-1.5B-Instruct-AWQ model served via SGLang.

---

## How It Works

**1. Document Ingestion**

A PDF is loaded, split into variable-length text chunks, embedded with `sentence-transformers`, and indexed in a FAISS vector store. Chunk size is tuned by the adaptive controller at query time based on estimated complexity.

**2. Query Analysis and Adaptive Retrieval**

Each query is scored for semantic complexity. The adaptive controller translates that score into a retrieval budget:

```
k = max(5, 2 + complexity * 18)   # between 5 and 20 chunks
hops = 2 if complexity > 0.6 else 1
```

High-complexity queries trigger up to three retrieval hops. After each hop, a confidence score is computed; retrieval continues only if confidence falls below 0.75 and the latency budget allows. Under time pressure the controller halves k automatically.

**3. Triton Cosine Similarity**

Vector similarity is computed by a hand-written Triton kernel. It tiles the query and document matrices across the GPU in 32x32 blocks with a 64-wide inner K dimension, accumulating dot products and squared norms simultaneously before writing normalized cosine scores. Falls back to PyTorch automatically on CPU or when Triton is unavailable.

**4. SGLang Inference with KV Prefix Caching**

Retrieved chunks are assembled into a prompt and sent to a locally hosted SGLang server. SGLang's radix attention implementation reuses computed KV states when the document prefix of a new prompt matches one already in cache, cutting time-to-first-token for repeated or clustered queries. In benchmarking, queries within the same topic cluster achieved a 38% cache hit rate.

---

## Components

| Path | Role |
|---|---|
| `src/kernels/cosine_similarity.py` | Triton block-tiled cosine kernel with PyTorch fallback |
| `src/adaptive/controller.py` | Complexity-to-retrieval-budget policy, hop controller |
| `src/retrieval/retriever.py` | Multi-hop retriever with confidence-based termination |
| `src/retrieval/chunker.py` | Adaptive text chunking |
| `src/retrieval/embedder.py` | Sentence embedding wrapper |
| `src/retrieval/reranker.py` | Cross-encoder reranker (enabled on high-complexity queries) |
| `src/agents/query_analyzer.py` | Complexity scoring agent |
| `src/agents/retrieval_agent.py` | Orchestrates multi-hop retrieval loop |
| `src/agents/generator_agent.py` | Calls SGLang streaming API, measures TTFT |
| `src/agents/critique_agent.py` | Confidence scoring on retrieved chunks |
| `src/caching/prefix_cache.py` | Tracks prefix hashes and TTFT to detect cache hits |
| `src/evaluation/ragas_eval.py` | RAGAS pipeline (faithfulness, relevancy, precision, recall) |

---

## Evaluation

Evaluation uses the [RAGAS](https://github.com/explodinggradients/ragas) framework across four metrics:

- **Faithfulness** - fraction of answer claims grounded in retrieved context
- **Answer Relevancy** - semantic alignment between the question and the answer
- **Context Precision** - fraction of retrieved chunks that are actually relevant
- **Context Recall** - coverage of ground-truth information in the retrieved set

The benchmark set is organized into 10 topic clusters with five questions each. Within a cluster, questions share a dominant keyword so the retriever consistently pulls the same chunks, exercising the prefix cache.

---

## Stack

| Layer | Technology |
|---|---|
| LLM Inference | SGLang (radix attention, KV prefix cache) |
| Model | Qwen2.5-1.5B-Instruct-AWQ (4-bit AWQ) |
| Vector Search | FAISS + custom Triton cosine kernel |
| Embeddings | sentence-transformers |
| Document Loading | PyMuPDF |
| Evaluation | RAGAS |
| GPU | CUDA 12.4, Triton 3.x |
