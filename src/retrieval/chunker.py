def chunk_text(text, doc_id, size=512, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i : i + size]
        chunks.append({
            "text":     " ".join(chunk_words),
            "doc_id":   doc_id,
            "chunk_id": chunk_id,
        })
        i += size - overlap
        chunk_id += 1
    return chunks


def adaptive_chunk_size(doc_length, query_complexity, default=512, min_size=128, max_size=2048):
    doc_factor        = min(1.5, doc_length / 5000)
    complexity_factor = 1.0 - 0.4 * query_complexity
    size = int(default * doc_factor * complexity_factor)
    return max(min_size, min(max_size, size))
