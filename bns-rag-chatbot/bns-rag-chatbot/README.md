---
title: BNS AI Assistant
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# ⚖️ BNS AI Assistant — Legal Document RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the **Bharatiya Nyaya Sanhita (BNS)** using only the text of the indexed documents, and cites the sections it used.

**Stack:** LangChain · ChromaDB · Hugging Face (sentence-transformers embeddings + Inference API LLM) · Gradio · Hugging Face Spaces

> General information only — not legal advice.

## How it works

```
BNS PDF ──► pypdf parsing ──► cleaning ──► section-aware chunking ──► HF embeddings ──► ChromaDB
                                                                                            │
question ──► embed ──► MMR similarity search (top-k) ◄─────────────────────────────────────┘
                              │
                              ▼
            prompt (context + strict grounding rules) ──► Hugging Face LLM ──► answer + sources
```

- **Ingestion pipeline (`ingest.py`)** — PDF parsing, text cleaning, chunking, embedding generation and vector indexing in one command.
- **Chunking strategy** — legal text is split on numbered-section and chapter boundaries first, then paragraphs/sentences (1000 chars, 150 overlap). Each chunk keeps `page`, `section` and `chapter` metadata so answers can cite sources.
- **Retrieval** — cosine similarity over normalised embeddings with MMR re-ranking to avoid near-duplicate overlapping chunks.
- **Hallucination control** — the prompt restricts the model to the retrieved context, requires section citations, and defines a fixed "not found" reply; temperature is 0.1.
- **Source-aware responses** — the UI lists the sections/pages each answer was based on.

## Project layout

| File | Purpose |
|---|---|
| `config.py` | All tunables (chunk size, models, top-k), overridable via env vars |
| `ingest.py` | PDF → chunks → embeddings → Chroma |
| `rag.py` | Retriever, prompt, LLM, `BNSAssistant.ask()` |
| `app.py` | Gradio chat UI (entry point for Spaces) |
| `evaluate.py` | Measure retrieval hit-rate, chunk-size comparison, latency |
| `tests/` | Offline tests (no downloads or API calls) |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then paste your Hugging Face token into .env
# put the official BNS PDF into ./data/  (from India Code / eGazette of India)

python ingest.py                # builds ./chroma_db (run once)
python app.py                   # opens the chat UI at http://127.0.0.1:7860
```

Create a free token at <https://huggingface.co/settings/tokens> (enable *"Make calls to Inference Providers"*).
If the default LLM is unavailable for your account, set `HF_LLM_REPO` to another chat model that Hugging Face serves.

## Deploy on Hugging Face Spaces

1. Create a new Space → SDK **Gradio**.
2. Push this repo to it (the `README.md` header above is the Space config).
3. Add the BNS PDF in `data/` (or commit a prebuilt `chroma_db/`, removing it from `.gitignore`).
4. Space **Settings → Secrets**: add `HUGGINGFACEHUB_API_TOKEN`.
5. On first start the app builds the index automatically if none exists.

## Testing & measuring

```bash
pytest tests -q                                   # offline unit/integration tests

cp eval_set.example.json eval_set.json            # add ~20-30 questions with the expected section numbers
python evaluate.py --k 4                          # hit@k + retrieval latency
python evaluate.py --k 4 --with-llm               # + end-to-end answer latency
python evaluate.py --compare-chunks 500 1000      # compare chunking settings
```

Use these scripts to produce the numbers you quote about the project (documents/chunks indexed, retrieval accuracy, response latency) — run them on your own index rather than relying on estimates.

## Limitations

- Scanned PDFs need OCR before ingestion.
- Answers are only as good as the retrieved context; the model may still occasionally misread a section.
- Section-number detection assumes the standard "103. Title." layout of the official PDF.
