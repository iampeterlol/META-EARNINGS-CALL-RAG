"""
parse_meta_transcript.py
Extract text from Meta earnings-call transcripts (PDF/TXT)
and convert to JSONL chunks for retrieval.
"""

import os, re, json, glob
from typing import List, Dict
from pdfminer.high_level import extract_text

RAW_DIR = "data_raw/META"
OUT_DIR = "data_parsed"

def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    lines = [ln.strip() for ln in s.split("\n")]
    return "\n".join(lines).strip()

def read_text(path: str) -> str:
    """Extract text from PDF or read TXT."""
    if path.lower().endswith(".pdf"):
        try:
            txt = extract_text(path) or ""
            return normalize_text(txt)
        except Exception as e:
            print(f"[warn] pdf read fail: {path} ({e})")
            return ""
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return normalize_text(f.read())

def split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)

def chunk_sentences(sents: List[str], min_words=100, max_words=200, overlap=30) -> List[str]:
    """Pack sentences into roughly min–max word chunks."""
    chunks, buf, wc = [], [], 0
    for s in sents:
        w = len(s.split())
        if wc + w <= max_words:
            buf.append(s); wc += w
        else:
            if wc < min_words:
                buf.append(s)
            chunks.append(" ".join(buf))
            tail = " ".join(" ".join(buf).split()[-overlap:])
            buf = [tail, s]; wc = len(" ".join(buf).split())
    if buf:
        chunks.append(" ".join(buf))
    return [c.strip() for c in chunks if len(c.split()) > 40]

def parse_one(path: str, year: int, quarter: str) -> List[Dict]:
    txt = read_text(path)
    sents = split_sentences(txt)
    chunks = chunk_sentences(sents)
    out = []
    for i, ch in enumerate(chunks, 1):
        prefix = f"[META | YEAR={year} | QUARTER={quarter}] "
        out.append({
            "chunk_id": f"META_{year}{quarter}_{i:04d}",
            "ticker": "META",
            "fiscal_year": year,
            "fiscal_quarter": quarter,
            "text": prefix + ch,   # inject prefix
            "source_file": os.path.basename(path)
        })
    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pdfs = glob.glob(os.path.join(RAW_DIR, "*.pdf"))
    print(f"[info] found {len(pdfs)} PDF files")

    for p in pdfs:
        m = re.search(r"Q([1-4])[-_']?(\d{2,4})", p)
        if not m:
            print(f"[skip] cannot parse quarter/year: {p}")
            continue
        q, y = f"Q{m.group(1)}", int(m.group(2))
        out_path = os.path.join(OUT_DIR, f"META_{y}{q}.jsonl")

        chunks = parse_one(p, y, q)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in chunks:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[done] {p} -> {len(chunks)} chunks -> {out_path}")

if __name__ == "__main__":
    main()