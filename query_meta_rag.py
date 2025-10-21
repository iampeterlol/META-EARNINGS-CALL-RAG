#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query_meta_rag.py
Simple retrieval demo over Meta earnings-call transcripts.
"""

import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "index/meta.index"
IDS_PATH   = "index/meta_ids.npy"
META_PATH  = "index/meta_meta.jsonl"

def load_meta():
    mp = {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            mp[obj["chunk_id"]] = obj
    return mp

def main():
    print("Loading model and index ...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    index = faiss.read_index(INDEX_PATH)
    ids = np.load(IDS_PATH, allow_pickle=True)
    meta_map = load_meta()

    while True:
        q = input("\nQuestion (or 'exit'): ").strip()
        if not q or q.lower() == "exit":
            break
        qv = model.encode([q], normalize_embeddings=True).astype("float32")
        D, I = index.search(qv, 5)
        print("\nTop-5 matches:\n")
        for rank, idx in enumerate(I[0], 1):
            cid = ids[idx]
            obj = meta_map[cid]
            snippet = obj["text"].replace("\n", " ")
            print(f"[{rank}] {obj['fiscal_quarter']} {obj['fiscal_year']} | {obj['source_file']}")
            print(snippet[:500] + " ...\n")

if __name__ == "__main__":
    main()