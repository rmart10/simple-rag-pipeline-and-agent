# Simple RAG Pipeline and Agent

A minimal, readable implementation of a Retrieval-Augmented Generation (RAG) pipeline using [Chroma](https://www.trychroma.com/), [Claude](https://www.anthropic.com/), and [Gradio](https://www.gradio.app/). Built as a companion to a blog post walking through how RAG works from scratch.

The demo uses a Chamberlain garage door opener manual as the source document — but the pipeline is generic and can be pointed at any PDF.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What is RAG?

Large language models are powerful but they only know what was in their training data. RAG solves this by giving the model a way to look things up at query time:

1. **Ingest** — split a document into chunks, embed each chunk as a vector, and store them in a vector database
2. **Retrieve** — when a question comes in, embed it and find the most similar chunks
3. **Generate** — pass those chunks to the LLM as context and let it answer grounded in the source material

This project implements that loop as a Claude tool-use agent: Claude decides when to call the `search_manual` tool, what to search for, and how to synthesize the results into an answer.

---

## Architecture

```
documents/
└── 114-6133-000.pdf          ← source document

ingest.py                      ← run once to build the vector store
  └── pypdf → chunks → Chroma (all-MiniLM-L6-v2 embeddings)

app.py                         ← the runtime agent + UI
  └── Gradio UI
      └── Claude (claude-sonnet-4-6)
          └── search_manual tool → Chroma → top 4 chunks → answer
```

**`ingest.py`** reads the PDF page by page, splits each page into 300-word overlapping chunks (50-word overlap), and stores them in a persistent Chroma collection. Chroma handles embeddings automatically using its built-in `all-MiniLM-L6-v2` ONNX model — no separate embedding API needed.

**`app.py`** runs a ReAct-style agent loop. Each user message triggers a call to the Anthropic API with the `search_manual` tool available. Claude retrieves relevant chunks, reasons over them, and returns a grounded answer with page citations. The Gradio `ChatInterface` wraps everything in a multi-turn chat UI.

---

## Setup

**Prerequisites:** Python 3.11+, an [Anthropic API key](https://console.anthropic.com/)

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/rmart10/simple-rag-pipeline-and-agent.git
cd simple-rag-pipeline-and-agent
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
# open .env and set ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

```bash
# Step 1 — ingest the PDF (run once)
python ingest.py

# Step 2 — start the chat UI
python app.py
# → open http://localhost:7860
```

On first run, `ingest.py` will download the `all-MiniLM-L6-v2` embedding model (~80 MB). Subsequent runs are instant.

---

## Using Your Own Document

1. Replace the PDF in `documents/`
2. Update `PDF_PATH` in `ingest.py`
3. Update the `SYSTEM` prompt and `TOOLS[0]["description"]` in `app.py` to describe the new subject
4. Re-run `ingest.py`

---

## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Claude API client |
| `chromadb` | Vector store + embeddings |
| `pypdf` | PDF text extraction |
| `gradio` | Chat UI |
| `python-dotenv` | Load API key from `.env` |
