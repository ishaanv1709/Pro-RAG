"""
Tracks real TTFT measurements from SGLang.

SGLang handles KV cache reuse automatically when --enable-prefix-caching is set.
We don't manage the cache ourselves — we just measure TTFT via streaming
and track whether repeated doc prefixes result in faster first-token times.
"""

import hashlib
import time

_seen_prefixes = {}   # hash -> first seen time
_ttft_records  = []   # list of {"hash": str, "ttft_ms": float, "cache_likely_hit": bool}


def _hash(doc_texts):
    combined = "\n---\n".join(sorted(doc_texts))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def record(doc_texts, ttft_ms):
    """
    Call after each SGLang generation.
    Returns True if this was likely a cache hit (repeated prefix, lower TTFT).
    """
    key        = _hash(doc_texts)
    first_time = key not in _seen_prefixes

    if first_time:
        _seen_prefixes[key] = {"first_ttft": ttft_ms, "seen_at": time.time()}

    baseline     = _seen_prefixes[key]["first_ttft"]
    cache_likely = (not first_time) and (ttft_ms < baseline * 0.75)

    _ttft_records.append({
        "prefix_hash":  key,
        "ttft_ms":      ttft_ms,
        "first_seen":   first_time,
        "cache_likely": cache_likely,
    })
    return cache_likely


def report():
    if not _ttft_records:
        return {"note": "No records yet"}

    first_hits   = [r["ttft_ms"] for r in _ttft_records if r["first_seen"]]
    repeat_hits  = [r["ttft_ms"] for r in _ttft_records if not r["first_seen"]]
    cache_likely = [r for r in _ttft_records if r["cache_likely"]]

    avg_first  = sum(first_hits)  / len(first_hits)  if first_hits  else 0
    avg_repeat = sum(repeat_hits) / len(repeat_hits) if repeat_hits else 0
    speedup    = avg_first / avg_repeat if avg_repeat > 0 else 0

    return {
        "total_requests":    len(_ttft_records),
        "unique_prefixes":   len(_seen_prefixes),
        "avg_ttft_first_ms": round(avg_first, 2),
        "avg_ttft_repeat_ms": round(avg_repeat, 2),
        "ttft_speedup":      round(speedup, 2),
        "likely_cache_hits": len(cache_likely),
    }
