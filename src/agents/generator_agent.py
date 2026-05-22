import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer using ONLY the provided context. Be concise and factual. "
    "If the context is insufficient, say so."
)


def _messages(query, chunks):
    context = "\n\n".join(f"[{i+1}] {c['chunk']['text']}" for i, c in enumerate(chunks))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


def generate_hf(query, chunks):
    """
    HuggingFace Inference API — plain LLM call, no prefix caching.
    Used by S1-S4 so they get the same model as SGLang but without any caching.
    """
    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=os.getenv("HF_TOKEN", ""),
    )
    model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    t0       = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=_messages(query, chunks),
        temperature=0.1,
        max_tokens=512,
    )
    total = (time.perf_counter() - t0) * 1000
    return {
        "answer":   response.choices[0].message.content,
        "ttft_ms":  total,   # HF API doesn't expose real TTFT
        "total_ms": total,
    }


def generate(query, chunks):
    """
    SGLang server — streaming for real TTFT measurement.
    Used by S5 (prefix caching enabled on the server).
    """
    port  = os.getenv("SGLANG_PORT",  "30000")
    model = os.getenv("SGLANG_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")

    t0     = time.perf_counter()
    ttft   = None
    answer = ""

    try:
        stream = client.chat.completions.create(
            model=model, messages=_messages(query, chunks),
            temperature=0.1, max_tokens=512, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                answer += delta
        total = (time.perf_counter() - t0) * 1000
        return {"answer": answer, "ttft_ms": ttft or total, "total_ms": total}

    except Exception as e:
        raise RuntimeError(
            f"SGLang server not reachable at port {port}. "
            f"Start it with: python -m sglang.launch_server "
            f"--model {model} --enable-prefix-caching --port {port}\n{e}"
        )
