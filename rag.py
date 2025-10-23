"""
rag.py
RAG pipeline using FAISS retrieval + Gemini summarization.
"""

from dotenv import load_dotenv
import os
import json, numpy as np, faiss, google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --------------- CONFIG ---------------
EMB_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_PATH = "index/meta.index"
IDS_PATH = "index/meta_ids.npy"
MAP_PATH = "index/meta_meta.jsonl"
TOP_K = 4
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
# --------------------------------------

def load_meta_map():
    mp = {}
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            mp[o["chunk_id"]] = o
    return mp

def build_context(retrieved_docs):
    sections = []
    for d in retrieved_docs:
        sec = f"[{d['fiscal_quarter']} {d['fiscal_year']}] {d['text']}"
        sections.append(sec)
    return "\n\n".join(sections)

def main():
    print("Loading model and index...")
    emb_model = SentenceTransformer(EMB_MODEL)
    index = faiss.read_index(INDEX_PATH)
    ids = np.load(IDS_PATH, allow_pickle=True)
    meta = load_meta_map()

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    while True:
        q = input("\nEnter your question (or 'exit'): ").strip()
        if not q or q.lower() == "exit":
            break

        qv = emb_model.encode([q], normalize_embeddings=True).astype("float32")
        D, I = index.search(qv, TOP_K)
        retrieved = [meta[ids[i]] for i in I[0]]
        context = build_context(retrieved)

        prompt = f"""
You are a financial analyst assistant.
Using the excerpts from Meta's earnings call transcripts below,
answer the question and cite which quarter(s) your evidence came from.

Context:
{context}

Question: {q}
"""

        print("Calling Gemini...")
        response = model.generate_content(prompt)
        print("\Gemini Answer:\n")
        print(response.text)
        print("\n" + "-" * 80)

if __name__ == "__main__":
    main()