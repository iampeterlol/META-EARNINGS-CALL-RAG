"""
build_faiss_index.py
Build a FAISS index from JSONL chunks.
"""

import os, json, glob, numpy as np, faiss
from sentence_transformers import SentenceTransformer

DATA_GLOB = "data_parsed/META_*.jsonl"
OUT_DIR = "index"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    texts, meta = [], []
    for path in glob.glob(DATA_GLOB):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
                meta.append(obj)

    print(f"[info] total {len(texts)} chunks to index")
    if not texts:
        return

    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype("float32")
    dim = emb.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(emb)

    faiss.write_index(index, os.path.join(OUT_DIR, "meta.index"))
    np.save(os.path.join(OUT_DIR, "meta_ids.npy"), np.array([m["chunk_id"] for m in meta], dtype=object))
    with open(os.path.join(OUT_DIR, "meta_meta.jsonl"), "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("[done] index built")

if __name__ == "__main__":
    main()