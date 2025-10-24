import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer

# Load index and meta
index = faiss.read_index("index/meta.index")
ids = np.load("index/meta_ids.npy", allow_pickle=True)
meta = {}
with open("index/meta_meta.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        meta[o["chunk_id"]] = o

print(f"Total chunks in index: {index.ntotal}, meta: {len(meta)}")

# Test retrieval
query = "revenue guidance Q1 2025"
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
qv = model.encode([query], normalize_embeddings=True).astype("float32")

D, I = index.search(qv, 10)

for j, i in enumerate(I[0]):
    print(f"\n[{meta[ids[i]]['fiscal_quarter']} {meta[ids[i]]['fiscal_year']}] (score={D[0][j]:.3f})")
    print(meta[ids[i]]["text"])