"""
Streamlit RAG Web App for Meta Earnings Call
Uses FAISS retrieval + Gemini summarization.
"""

import os, json, numpy as np, faiss, streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import re

# ========== CONFIG ==========
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMB_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_PATH = "index/meta.index"
IDS_PATH = "index/meta_ids.npy"
MAP_PATH = "index/meta_meta.jsonl"
TOP_K = 10
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

# ======= Universal constraint parser =======
CAPEX_SYNS = [
    "capex", "capital expenditure", "capital expenditures",
    "capital investment", "data center", "datacenter",
    "servers", "infrastructure", "ai infrastructure",
    "network", "gpu", "training clusters"
]
def parse_constraints(q: str):
    """Extract generic constraints (years/quarters/keywords) from question."""
    ql = q.lower()
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", ql)]
    qtr = re.findall(r"\b(q[1-4])\b", ql, flags=re.I)
    fiscal = re.findall(r"\bfy\s*(\d{2})\b", ql, flags=re.I)
    if fiscal:
        years += [2000 + int(fiscal[-1])]  # FY23 -> 2023

    keywords = set()
    for w in CAPEX_SYNS:
        if w in ql:
            keywords.add(w)
    # COVID/e-commerce detection
    if "covid" in ql or "pandemic" in ql:
        keywords |= {"covid", "pandemic"}
    if "commerce" in ql or "e-commerce" in ql or "shop" in ql:
        keywords |= {"commerce", "shops"}

    return {"years": sorted(set(years)),
            "quarters": [x.upper() for x in qtr],
            "keywords": keywords}

def infer_quarter_year(r: dict) -> str:
    """Try to infer fiscal quarter/year (like Q12024) from metadata or filename."""
    fq = str(r.get("fiscal_quarter", "")).upper().strip()
    fy = str(r.get("fiscal_year", "")).strip()
    if fq and fy:
        return f"{fq}{fy}".replace(" ", "")

    # 从文件名、路径或 chunk 内容里推断
    candidates = " ".join([
        str(r.get("source", "")),
        str(r.get("file", "")),
        str(r.get("title", "")),
        str(r.get("path", "")),
        str(r.get("chunk", ""))[:500],
        str(r.get("text", ""))[:500],
    ]).upper()

    norm = candidates
    norm = re.sub(r'FIRST\s+QUARTER|QUARTER\s*1|QTR\s*1|\b1Q\b', 'Q1', norm)
    norm = re.sub(r'SECOND\s+QUARTER|QUARTER\s*2|QTR\s*2|\b2Q\b', 'Q2', norm)
    norm = re.sub(r'THIRD\s+QUARTER|QUARTER\s*3|QTR\s*3|\b3Q\b', 'Q3', norm)
    norm = re.sub(r'FOURTH\s+QUARTER|QUARTER\s*4|QTR\s*4|\b4Q\b', 'Q4', norm)

    m = re.search(r'(Q[1-4])\D{0,6}(20\d{2})', norm)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    m = re.search(r'(20\d{2})\D{0,6}(Q[1-4])', norm)
    if m:
        return f"{m.group(2)}{m.group(1)}"

    return ""

def query_rag(question, model, index, ids, meta):
    """Generic RAG retrieval with dynamic constraints (year/quarter/topic)."""
    index.hnsw.efSearch = 256
    TOP_K_RAW = 400
    TOP_K_FINAL = 10

    cons = parse_constraints(question)
    years, quarters, kwds = set(cons["years"]), set(cons["quarters"]), cons["keywords"]

    # --- inject constraints into semantic query
    boost = question
    if years:
        boost += " [focus on " + ", ".join(map(str, years)) + "]"
    if quarters:
        boost += " [focus on " + ", ".join(quarters) + "]"
    if kwds:
        boost += " [keywords: " + ", ".join(sorted(kwds)) + "]"

    qv = model.encode([boost], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, TOP_K_RAW)
    cands = [meta[ids[i]] | {"score": float(D[0][j])} for j, i in enumerate(I[0])]

    # --- soft filter by constraints
    def infer_year(r):
        y = r.get("fiscal_year")
        if isinstance(y, int): return y
        try:
            return int(str(y)[:4])
        except: return None

    def infer_quarter(r):
        q = str(r.get("fiscal_quarter", "")).upper()
        m = re.search(r"Q[1-4]", q)
        return m.group(0) if m else None

    filtered = []
    for r in cands:
        y, q = infer_year(r), infer_quarter(r)
        text = (r.get("text") or "").lower()
        score = r["score"]

        kw_bonus = sum(1 for k in kwds if k in text) * 0.1 if kwds else 0
        y_bonus = 0.4 if (y in years and years) else 0
        q_bonus = 0.3 if (q in quarters and quarters) else 0

        final_score = score + kw_bonus + y_bonus + q_bonus
        r["final_score"] = final_score
        filtered.append(r)

    filtered.sort(key=lambda x: x["final_score"], reverse=True)
    retrieved = filtered[:TOP_K_FINAL]

    # if strict constraints (year/quarter) exist and none match → say unavailable
    if (years or quarters) and not any(
        (infer_year(r) in years if years else True)
        and (infer_quarter(r) in quarters if quarters else True)
        for r in retrieved
    ):
        return "Sorry, the requested year/quarter information is not present in the indexed transcripts.", cands

    # build context for LLM
    context = build_context(retrieved)
    prompt = f"""
        You are a financial analyst assistant.
        Use only the excerpts below. If the answer isn't in them, say so.

        Context:
        {context}

        Question: {question}
        """
    gemini = genai.GenerativeModel(GENAI_MODEL)
    resp = gemini.generate_content(prompt)
    return resp.text, retrieved


# ================= Streamlit UI =================
st.set_page_config(page_title="Meta Earnings Call Analyst", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings & Info")
    st.markdown(f"**Embedding Model:** `{EMB_MODEL}`")
    st.markdown(f"**LLM Model:** `{GENAI_MODEL}`")
    st.markdown("---")

    if st.button("🧹 Clear chat"):
        st.session_state["history"] = []
        st.experimental_rerun()

    show_context = st.checkbox("Show retrieved context", value=True)
    save_logs = st.checkbox("Auto-save chats to file", value=True)
    st.markdown("---")
    st.caption("💡 Example: *“What were Meta’s capex priorities in 2023?”*")

# ---------- Session init ----------
if "history" not in st.session_state:
    st.session_state["history"] = []  # stores [{"role": "user"/"assistant", "content": str, "retrieved": list}]

st.title("💬 Meta Earnings Call Analyst")

# ---------- Display existing conversation ----------
for chat in st.session_state["history"]:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# ---------- Handle new user input ----------
if prompt := st.chat_input("Ask a question about Meta’s earnings calls..."):

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["history"].append({"role": "user", "content": prompt, "retrieved": []})

    # Combine recent assistant replies as context for multi-turn RAG
    past_context = "\n\n".join(
        f"Q: {h['content']}\nA: {h['content']}"
        for h in st.session_state["history"][-4:]  # last 4 turns
        if h["role"] == "assistant"
    )

    full_query = prompt
    if past_context:
        full_query = f"Previous conversation:\n{past_context}\n\nUser's new question:\n{prompt}"

    # Retrieve and answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and summarizing with Gemini..."):
            model, index, ids, meta = load_resources()
            try:
                answer, retrieved = query_rag(full_query, model, index, ids, meta)
            except Exception as e:
                st.error(f"⚠️ Retrieval or LLM error: {e}")
                answer, retrieved = "Sorry, something went wrong.", []

        st.markdown(answer)
        st.session_state["history"].append({
            "role": "assistant",
            "content": answer,
            "retrieved": retrieved
        })

        # ----- save to disk -----
        if save_logs:
            import json, datetime, os
            os.makedirs("chat_logs", exist_ok=True)
            log_path = "chat_logs/meta_chatlog.jsonl"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "user_query": prompt,
                    "assistant_answer": answer,
                    "retrieved_chunks": [
                        {
                            "file": d.get("source_file"),
                            "fiscal_year": d.get("fiscal_year"),
                            "fiscal_quarter": d.get("fiscal_quarter"),
                            "score": d.get("score", 0)
                        }
                        for d in retrieved
                    ]
                }, ensure_ascii=False) + "\n")

        # ----- Show retrieved context -----
        if show_context and retrieved:
            with st.expander("📚 Retrieved Context"):
                for i, doc in enumerate(retrieved, 1):
                    st.caption(f"Relevance Score: {doc.get('score', 0):.2f}")
                    st.markdown(f"**{i}. {doc['fiscal_quarter']} {doc['fiscal_year']}** — *{doc['source_file']}*")
                    st.write(re.sub(r'\$', r'\\$', doc.get('text', '')))
                    st.markdown("---")