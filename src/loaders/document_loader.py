from pathlib import Path


def load_pdf(file_path):
    try:
        import fitz
    except ImportError:
        raise ImportError("Run: pip install pymupdf")

    doc   = fitz.open(str(file_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({
                "id":   f"{Path(file_path).stem}_p{i+1}",
                "text": text,
            })
    doc.close()
    return pages


def load_txt(file_path):
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [{"id": Path(file_path).stem, "text": text}]


def load_file(file_path):
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {file_path}")
    if p.suffix.lower() == ".pdf":
        docs = load_pdf(p)
    elif p.suffix.lower() == ".txt":
        docs = load_txt(p)
    else:
        raise ValueError(f"Unsupported: {p.suffix}  (use .pdf or .txt)")
    print(f"Loaded {len(docs)} section(s) from {p.name}")
    return docs


def load_dir(dir_path):
    d    = Path(dir_path)
    docs = []
    for ext in [".pdf", ".txt"]:
        for f in sorted(d.glob(f"**/*{ext}")):
            try:
                docs.extend(load_file(f))
            except Exception as e:
                print(f"  Skipped {f.name}: {e}")
    print(f"Total: {len(docs)} sections from {dir_path}")
    return docs
