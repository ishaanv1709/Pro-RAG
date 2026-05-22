import re

HIGH_COMPLEXITY = ["compare", "contrast", "difference", "why", "how does",
                   "explain", "analyze", "relationship", "implications"]
LOW_COMPLEXITY  = ["what is", "who is", "when was", "where is", "define",
                   "list", "how many", "which year"]

MULTIHOP_PATTERNS = [r"\band\b", r"\balso\b", r"\bfurthermore\b",
                     r"\bfirst.*then\b", r"\bboth\b"]


def analyze_query(query):
    q = query.lower()

    high = sum(1 for s in HIGH_COMPLEXITY if s in q)
    low  = sum(1 for s in LOW_COMPLEXITY  if s in q)
    word_factor = min(1.0, len(query.split()) / 30)

    complexity = float(max(0.0, min(1.0, 0.3 + high * 0.25 - low * 0.15 + word_factor * 0.3)))

    multihop_hits = sum(1 for p in MULTIHOP_PATTERNS if re.search(p, q))

    if multihop_hits >= 2 or complexity > 0.7:
        query_type    = "multi_hop"
        estimated_hops = min(3, 1 + multihop_hits)
    elif complexity < 0.35:
        query_type    = "factoid"
        estimated_hops = 1
    else:
        query_type    = "abstract"
        estimated_hops = 2

    sub_queries = []
    if query_type == "multi_hop":
        parts = re.split(r"\band\b|\balso\b|\bfurthermore\b", query, flags=re.I)
        sub_queries = [p.strip().rstrip("?.") for p in parts if len(p.strip()) > 10][:3]

    return {
        "query":       query,
        "complexity":  complexity,
        "type":        query_type,
        "sub_queries": sub_queries,
        "hops":        estimated_hops,
    }
