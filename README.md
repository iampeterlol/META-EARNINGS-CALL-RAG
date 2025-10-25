# META Earnings Call RAG

This project is an **end-to-end Retrieval-Augmented Generation (RAG) system** for analyzing and querying **Meta (Facebook) earnings call transcripts**.  
It used **FAISS vector indexing**, **LLM-based question answering**, and a simple **Streamlit web interface** for interactive exploration.

---

## 🧠 Project Overview

The pipeline extracts and processes Meta's quarterly earnings call transcripts, builds a semantic vector index using FAISS, and enables users to ask natural-language questions (e.g., “What’s Meta’s total revenue in Q1 2024?”).  
Relevant transcript segments are retrieved and summarized by an LLM, producing concise, context-aware answers.

![alt text](<Earnings call analyst.png>)
---

## 📂 Project Structure
```
META-EARNINGS-CALL-RAG/
├── chat_logs/                # JSONL logs of interactive chat sessions
├── data_raw/                 # Raw transcript data (as downloaded or scraped)
├── data_parsed/              # Parsed and cleaned text segments
├── index/                    # FAISS index and vector embeddings
├── .env                      # Environment variables (API keys, etc.)
├── .gitignore
├── app.py                    # Streamlit front-end for user interaction
├── build_faiss_index.py      # Script to build FAISS index from parsed data
├── index_test.py             # Test utility for verifying index retrieval
├── parse_meta_transcript.py  # Script to clean & segment raw transcripts
├── query_meta_rag.py         # Query interface combining retriever + LLM
├── rag.py                    # Core RAG logic (retrieval + generation)
└── README.md
```
## 🧩 Workflow
In your terminal:
### 1️⃣ Parse Transcripts
```
python parse_meta_transcript.py
```
### 2️⃣ Build FAISS Index
```
python build_faiss_index.py
```
### 3️⃣ Launch Streamlit App
```
streamlit run app.py
```

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| Language Model | `gemini-2.0-flash` |
| Embeddings | `BAAI/bge-base-en-v1.5` |
| Vector Database | FAISS |
| Front-End | Streamlit |
| Data Processing | Python, Pandas, JSON |
| Environment | Conda / venv |