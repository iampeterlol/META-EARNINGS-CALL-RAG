"""
Streamlit RAG Web App for Meta Earnings Call
Uses FAISS retrieval + Gemini summarization.
"""

import os, json, numpy as np, faiss, streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import re
import datetime

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

def summarize_history_for_context(history, max_turns=4):
    """
    Build a concise conversation summary from the last few turns.
    Returns text like:
    'User previously asked about ... Gemini replied that ...'
    """
    if not history:
        return ""
    relevant = [h for h in history if h["role"] in ("user", "assistant")][-max_turns*2:]
    pairs = []
    q = None
    for h in relevant:
        if h["role"] == "user":
            q = h["content"]
        elif h["role"] == "assistant" and q:
            pairs.append((q, h["content"]))
            q = None
    summary_lines = []
    for i, (u, a) in enumerate(pairs, 1):
        summary_lines.append(f"Turn {i}: User asked '{u}' → Assistant answered '{a[:200]}...'")
    return "\n".join(summary_lines)

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

def query_rag(question, model, index, ids, meta, conversation_summary=""):
    """Generic RAG retrieval with dynamic constraints (year/quarter/topic) and optional conversation context summary."""
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

    # === 加入对话摘要到查询向量 ===
    if conversation_summary:
        boost = f"Previous conversation:\n{conversation_summary}\n\nUser's new question:\n{boost}"

    qv = model.encode([boost], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, TOP_K_RAW)
    cands = [meta[ids[i]] | {"score": float(D[0][j])} for j, i in enumerate(I[0])]

    # --- soft filter by constraints
    def infer_year(r):
        y = r.get("fiscal_year")
        if isinstance(y, int):
            return y
        try:
            return int(str(y)[:4])
        except:
            return None

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

    # === 构建包含上下文的 LLM prompt ===
    context = build_context(retrieved)
    prompt = f"""
        You are a financial analyst assistant.
        Use only the excerpts below. If the answer isn't in them, say so.

        Conversation so far:
        {conversation_summary if conversation_summary else "(no prior context)"}

        Context:
        {context}

        Question: {question}
        """
    gemini = genai.GenerativeModel(GENAI_MODEL)
    resp = gemini.generate_content(prompt)
    return resp.text, retrieved

import datetime

# ========== Chat session utilities ==========
def list_chat_sessions():
    """List all saved chat sessions."""
    os.makedirs("chat_logs", exist_ok=True)
    files = sorted(
        [f for f in os.listdir("chat_logs") if f.endswith(".json")],
        key=lambda x: os.path.getmtime(os.path.join("chat_logs", x)),
        reverse=True
    )
    return files

def load_chat(filename):
    """Load a chat session."""
    path = os.path.join("chat_logs", filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_chat(filename, title, history):
    """Save a chat session."""
    path = os.path.join("chat_logs", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"title": title, "history": history}, f, ensure_ascii=False, indent=2)

def new_chat_filename(title="New Chat"):
    """Generate a unique filename."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = title.replace(" ", "_")[:40]
    return f"chat_{ts}_{safe_title}.json"

def generate_chat_title(prompt):
    """Generate a concise title (remove extra 'options', punctuation, etc.)."""
    try:
        gemini = genai.GenerativeModel(GENAI_MODEL)
        msg = f"Generate a single short, specific title (4–6 words) summarizing this question: '{prompt}'. Do not include 'Here are options' or multiple bullets."
        resp = gemini.generate_content(msg)
        raw_title = resp.text.strip()

        # 清理换行、冒号、列表符号等
        title = re.split(r'[\n:*•\-–]', raw_title)[0]
        title = re.sub(r'["*_`]+', '', title)  # 去除符号
        title = re.sub(r'\s+', ' ', title).strip()
        title = title.replace("Here are a few options", "").strip()

        # 限制长度并大写首字母
        if len(title) > 80:
            title = title[:80]
        title = title[:1].upper() + title[1:] if title else "New Chat"
        return title or "New Chat"

    except Exception as e:
        print(f"[warn] title generation failed: {e}")
        return "New Chat"

# ================= Streamlit UI =================
st.set_page_config(page_title="Meta Earnings Call Analyst", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🗂️ Chat Sessions")

    sessions = list_chat_sessions()
    if "current_chat" not in st.session_state:
        st.session_state["current_chat"] = sessions[0] if sessions else None
    if "chat_title" not in st.session_state:
        st.session_state["chat_title"] = "New Chat"
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # selected_chat = st.selectbox("Select chat session", options=[""] + sessions, index=(sessions.index(st.session_state["current_chat"]) + 1) if st.session_state["current_chat"] in sessions else 0)
    # Build mapping of filename to title (for cleaner dropdown display)
    chat_titles = {}
    for f in sessions:
        try:
            data = load_chat(f)
            title = data.get("title", f)
            # 简化标题显示（去掉 chat_时间戳_ 之类）
            clean_title = re.sub(r"^chat_\d{8}_\d{6}_", "", f)
            chat_titles[f] = title or clean_title
        except Exception:
            chat_titles[f] = f

    # 构建下拉选项显示
    options_display = [""] + [chat_titles[f] for f in sessions]
    selected_display = st.selectbox("Select chat session", options=options_display)

    # 找到被选中的文件名
    if selected_display:
        selected_chat = [k for k, v in chat_titles.items() if v == selected_display][0]
    else:
        selected_chat = ""

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("➕ New Chat"):
            st.session_state["history"] = []
            st.session_state["current_chat"] = None
            st.session_state["chat_title"] = "New Chat"
            st.experimental_rerun()
    with col2:
        if st.button("💾 Save Chat"):
            if st.session_state["current_chat"]:
                save_chat(st.session_state["current_chat"], st.session_state.get("chat_title", "New Chat"), st.session_state["history"])
                st.success(f"Chat saved as {st.session_state['current_chat']}")
            else:
                # save as new file
                filename = new_chat_filename(st.session_state.get("chat_title", "New Chat"))
                save_chat(filename, st.session_state.get("chat_title", "New Chat"), st.session_state["history"])
                st.session_state["current_chat"] = filename
                st.success(f"Chat saved as {filename}")
    with col3:
        if st.button("🗑️ Delete Chat"):
            if st.session_state["current_chat"]:
                try:
                    os.remove(os.path.join("chat_logs", st.session_state["current_chat"]))
                    st.success(f"Deleted {st.session_state['current_chat']}")
                    st.session_state["current_chat"] = None
                    st.session_state["history"] = []
                    st.session_state["chat_title"] = "New Chat"
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to delete chat: {e}")

    if selected_chat != "" and selected_chat != st.session_state.get("current_chat"):
        # Load selected chat
        data = load_chat(selected_chat)
        st.session_state["history"] = data.get("history", [])
        st.session_state["chat_title"] = data.get("title", "New Chat")
        st.session_state["current_chat"] = selected_chat
        st.experimental_rerun()

    show_context = st.checkbox("Show retrieved context", value=True)
    save_logs = st.checkbox("Auto-save chats to file", value=True)
    st.markdown("---")
    st.caption("💡 Example: *“What were Meta’s capex priorities in 2023?”*")
    st.markdown("---")

    st.header("⚙️ Settings & Info")
    st.markdown(f"**Embedding Model:** `{EMB_MODEL}`")
    st.markdown(f"**LLM Model:** `{GENAI_MODEL}`")

# ---------- Session init ----------
if "history" not in st.session_state:
    st.session_state["history"] = []  # stores [{"role": "user"/"assistant", "content": str, "retrieved": list}]
if "chat_title" not in st.session_state:
    st.session_state["chat_title"] = "New Chat"

st.title(f"💬 Meta Earnings Call Analyst")

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
                # answer, retrieved = query_rag(full_query, model, index, ids, meta)
                answer, retrieved = query_rag(prompt, model, index, ids, meta,
                                             conversation_summary=summarize_history_for_context(st.session_state["history"]))
            except Exception as e:
                st.error(f"Retrieval or LLM error: {e}")
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

        # ----- Auto-generate title and save current chat -----
        if st.session_state["chat_title"] == "New Chat" and st.session_state["history"]:
            # generate title from first user prompt
            first_user_prompt = next((h["content"] for h in st.session_state["history"] if h["role"] == "user"), None)
            if first_user_prompt:
                title = generate_chat_title(first_user_prompt)
                st.session_state["chat_title"] = title

        if st.session_state.get("current_chat"):
            save_chat(st.session_state["current_chat"], st.session_state.get("chat_title", "New Chat"), st.session_state["history"])
        else:
            # Save as new file
            filename = new_chat_filename(st.session_state.get("chat_title", "New Chat"))
            save_chat(filename, st.session_state.get("chat_title", "New Chat"), st.session_state["history"])
            st.session_state["current_chat"] = filename

        # ----- Show retrieved context -----
        if show_context and retrieved and "not present in the indexed transcripts" not in answer.lower():
            with st.expander("📚 Retrieved Context"):
                for i, doc in enumerate(retrieved, 1):
                    st.caption(f"Relevance Score: {doc.get('score', 0):.2f}")
                    st.markdown(f"**{i}. {doc['fiscal_quarter']} {doc['fiscal_year']}** — *{doc['source_file']}*")
                    st.write(re.sub(r'\$', r'\\$', doc.get('text', '')))
                    st.markdown("---")