# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Commands

```bash
# Step 1 — chunk and embed the PDF into Chroma (run once, or after changing the PDF)
.venv/bin/python ingest.py

# Step 2 — start the Gradio UI
.venv/bin/python app.py
# → http://localhost:7860
```

## Architecture

Two-script pipeline:

**`ingest.py`** — one-time ingestion step. Reads `documents/114-6133-000.pdf` with `pypdf`, splits each page into 300-word overlapping chunks (50-word overlap), and stores them in a persistent Chroma collection (`chroma_db/`, collection name `"manual"`). Chroma uses its built-in `all-MiniLM-L6-v2` ONNX model for embeddings — no separate embedding API needed. Re-running wipes and recreates the collection.

**`app.py`** — the runtime agent. At startup it connects to the existing Chroma collection and initialises an `anthropic.Anthropic()` client (key loaded from `.env` via `python-dotenv`). Each chat turn runs a ReAct-style tool-use loop:
1. Claude decides to call `search_manual(query)` → Chroma cosine-similarity search returns the top 4 chunks with page numbers
2. Chunks are fed back as `tool_result` messages
3. Claude generates a grounded answer citing page numbers
4. Loop exits on `stop_reason == "end_turn"`

Gradio 6 passes history as dicts with extra fields; the history conversion in `run_agent` strips everything except `role` and `content` before sending to the Anthropic API.

## Key constants (both files share the same values)

| Constant | File | Value |
|---|---|---|
| `CHROMA_PATH` | both | `"chroma_db"` |
| `COLLECTION_NAME` | both | `"manual"` |
| `MODEL` | app.py | `"claude-sonnet-4-6"` |
| `N_RESULTS` | app.py | `4` chunks per query |
| `CHUNK_SIZE` | ingest.py | `300` words |
| `CHUNK_OVERLAP` | ingest.py | `50` words |

## Swapping the document

1. Replace the PDF in `documents/`
2. Update `PDF_PATH` in `ingest.py`
3. Update the `SYSTEM` prompt and `TOOLS[0]["description"]` in `app.py` to reflect the new subject matter
4. Re-run `ingest.py`
