"""
Streamlit RAG Web App for Meta Earnings Call
Uses FAISS retrieval + Gemini summarization.
"""

import os, json, numpy as np, faiss, streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

# ========== CONFIG ==========
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMB_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_PATH = "index/meta.index"
IDS_PATH = "index/meta_ids.npy"
MAP_PATH = "index/meta_meta.jsonl"
TOP_K = 4
GENAI_MODEL = "gemini-2.0-flash"
# ============================


@st.cache_resource(show_spinner=False)
def load_resources():
    """Load embedding model, FAISS index, and metadata (cached)."""
    model = SentenceTransformer(EMB_MODEL)
    index = faiss.read_index(INDEX_PATH)
    ids = np.load(IDS_PATH, allow_pickle=True)
    meta = {}
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            meta[o["chunk_id"]] = o
    return model, index, ids, meta


def build_context(retrieved_docs):
    """Combine retrieved chunks for LLM context."""
    sections = []
    for d in retrieved_docs:
        sec = f"[{d['fiscal_quarter']} {d['fiscal_year']}] {d['text']}"
        sections.append(sec)
    return "\n\n".join(sections)


def query_rag(question, model, index, ids, meta):
    """Retrieve relevant chunks and call Gemini for summary."""
    qv = model.encode([question], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, TOP_K)
    retrieved = [meta[ids[i]] for i in I[0]]

    context = build_context(retrieved)

    prompt = f"""
        You are a financial analyst assistant.
        Based on the following Meta earnings call transcripts,
        answer the question clearly and cite which quarter(s) the statements came from.

        Context:
        {context}

        Question: {question}
        """

    gemini = genai.GenerativeModel(GENAI_MODEL)
    print(prompt)
    response = gemini.generate_content(prompt)

    return response.text, retrieved


# ================= Streamlit UI =================
st.set_page_config(page_title="Meta Earnings Call RAG", layout="wide")
st.title("💬 Meta Earnings Call RAG (Gemini-powered)")

question = st.text_input("Enter your question:", placeholder="e.g. What did Meta say about AI investment in 2024?")
run = st.button("🔍 Search")

if run and question.strip():
    with st.spinner("Retrieving and summarizing with Gemini..."):
        model, index, ids, meta = load_resources()
        answer, retrieved = query_rag(question, model, index, ids, meta)

    st.markdown("### 🧠 Gemini Answer")
    st.write(answer)

    with st.expander("📚 Retrieved Context"):
        for i, doc in enumerate(retrieved, 1):
            st.markdown(f"**{i}. {doc['fiscal_quarter']} {doc['fiscal_year']}** — *{doc['source_file']}*")
            st.write(doc["text"])
            st.markdown("---")