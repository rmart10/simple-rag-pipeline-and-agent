# Code Walkthrough & Deployment Checklist

## `ingest.py` — Building the Vector Database

This script runs once to transform the raw PDF into a searchable vector store. It never needs to run again unless the document changes.

---

### Constants

```python
PDF_PATH = "documents/114-6133-000.pdf"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "manual"
CHUNK_SIZE = 300   # words per chunk
CHUNK_OVERLAP = 50  # words of overlap between chunks
```

These four values are the only knobs you'd turn to adapt this pipeline to a different document. `CHUNK_SIZE` and `CHUNK_OVERLAP` control the granularity of retrieval — smaller chunks are more precise but lose surrounding context; larger chunks preserve context but may dilute relevance scores.

---

### `load_pdf` — Extracting Text by Page

```python
def load_pdf(path: str) -> list[dict]:
    reader = pypdf.PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages
```

`pypdf` reads each page and extracts its text content. Pages that yield no text (e.g. image-only pages) are silently dropped. Each page that does have text is stored as a dict with a 1-based `page` number — this gets carried through to Chroma metadata and surfaced in the agent's answers as a citation.

---

### `make_chunks` — Splitting Pages into Overlapping Chunks

```python
def make_chunks(pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        words = page["text"].split()
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for start in range(0, len(words), step):
            chunk_words = words[start : start + CHUNK_SIZE]
            if len(chunk_words) < 20:
                continue
            chunks.append({"text": " ".join(chunk_words), "page": page["page"]})
    return chunks
```

Text is split on whitespace into words, then a sliding window of `CHUNK_SIZE` words advances by `step = CHUNK_SIZE - CHUNK_OVERLAP` words each iteration. This means consecutive chunks share 50 words of context, preventing a sentence that straddles a chunk boundary from being missed entirely. Fragments shorter than 20 words (typically the tail of a page) are skipped.

---

### `main` — Embedding and Storing in Chroma

```python
client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass
collection = client.create_collection(COLLECTION_NAME)
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=[c["text"] for c in chunks],
    metadatas=[{"page": c["page"]} for c in chunks],
)
```

`PersistentClient` writes the database to disk at `chroma_db/` so the embeddings survive process restarts. The collection is deleted and recreated on every run so re-ingestion is always a clean slate. Chroma's default embedding function (`all-MiniLM-L6-v2`, downloaded on first run) converts each chunk's text to a vector automatically — no separate embedding API call is needed.

---

## `app.py` — The RAG Agent

This is the runtime. It loads the pre-built vector store and wires together a retrieval tool, a Claude agent loop, and a Gradio chat UI.

---

### Startup and Clients

```python
load_dotenv()

chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_collection(COLLECTION_NAME)
claude = anthropic.Anthropic()
```

`load_dotenv()` reads `.env` and injects `ANTHROPIC_API_KEY` into the environment before the `Anthropic()` client is constructed — so the key never has to be passed explicitly. The Chroma collection is opened read-only at startup; all disk I/O for retrieval happens inside `collection.query()`.

---

### Tool Definition

```python
TOOLS = [{
    "name": "search_manual",
    "description": "Search the Chamberlain garage door opener manual ...",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", ...}},
        "required": ["query"],
    },
}]
```

This JSON schema is sent to Claude with every request. It tells the model that a `search_manual` tool is available and what argument it expects. Claude decides autonomously whether to call it, what query string to pass, and whether to call it again with a refined query. The description is deliberately specific about the manual's topic areas so Claude calls the tool before attempting to answer from training data.

---

### System Prompt

```python
SYSTEM = (
    "You are a knowledgeable assistant for the Chamberlain garage door opener "
    "manual ... Always call search_manual before answering so your response is "
    "grounded in the manual. Cite the page number when relevant. ..."
)
```

Two behaviours are enforced here: always retrieve before answering (reducing hallucination), and always cite the page number (giving users a way to verify). The explicit instruction to say "the manual does not contain the answer" when retrieval comes up empty prevents the model from filling gaps with invented facts.

---

### `search_manual` — Retrieval Function

```python
def search_manual(query: str) -> str:
    results = collection.query(query_texts=[query], n_results=N_RESULTS)
    sections = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        sections.append(f"[Page {meta['page']}]\n{doc}")
    return "\n\n---\n\n".join(sections)
```

This is the bridge between Claude's tool call and Chroma. `collection.query()` embeds the query string with the same model used at ingest time and returns the `N_RESULTS` most similar chunks by cosine distance. The result is assembled into a plain-text string with page-number labels, which Claude receives as the `tool_result` content.

---

### `run_agent` — The Agent Loop

```python
def run_agent(user_message: str, history: list) -> str:
    messages = []
    for item in history:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            human, assistant = item
            ...
    messages.append({"role": "user", "content": user_message})

    while True:
        response = claude.messages.create(model=MODEL, ..., messages=messages)

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if hasattr(b, "text")), ...)

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = search_manual(block.input["query"])
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
```

This is the heart of the agent. The loop runs until Claude signals it is done:

- **`end_turn`** — Claude has produced its final text answer. Extract and return it.
- **`tool_use`** — Claude wants to call `search_manual`. Execute the retrieval, package the results as `tool_result` messages, append both the assistant turn and the results to the conversation, and loop back to let Claude continue reasoning.

The history conversion at the top strips Gradio 6's extra message fields (e.g. `metadata`) down to just `role` and `content`, which is all the Anthropic API accepts.

---

### Gradio Interface

```python
demo = gr.ChatInterface(
    fn=chat,
    title="Garage Door Opener Manual Assistant",
    description="...",
    examples=[...],
)

if __name__ == "__main__":
    demo.launch()
```

`ChatInterface` wraps the `chat` function in a full multi-turn chat UI with no additional code. The `examples` list populates clickable starter questions. `demo.launch()` starts a local web server on port 7860.

---

## Deployment Checklist

Use this checklist to get the project running in a new environment.

### Prerequisites

- [ ] Python 3.11 or later
- [ ] An [Anthropic API key](https://console.anthropic.com/)
- [ ] The PDF document placed in `documents/`

### First-time Setup

- [ ] Create and activate a virtual environment
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  ```
- [ ] Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
  > Note: Chroma will download the `all-MiniLM-L6-v2` embedding model (~80 MB) on first run of `ingest.py`.

- [ ] Create your `.env` file
  ```bash
  cp .env.example .env
  # then open .env and set ANTHROPIC_API_KEY=sk-ant-...
  ```

### Ingest the Document

- [ ] Run the ingestion script (only needed once, or after changing the PDF)
  ```bash
  python ingest.py
  ```
  Expected output:
  ```
  Loading documents/114-6133-000.pdf …
    19 pages extracted
    39 chunks created
    Done — 39 chunks stored in 'chroma_db'
  ```

### Run the App

- [ ] Start the Gradio server
  ```bash
  python app.py
  ```
- [ ] Open [http://localhost:7860](http://localhost:7860) in your browser
- [ ] Ask a test question (e.g. "How do I program a new remote?") and confirm you get a response with a page citation

### Adapting to a Different Document

- [ ] Replace the PDF in `documents/` and update `PDF_PATH` in `ingest.py`
- [ ] Update the `SYSTEM` prompt in `app.py` to describe the new subject matter
- [ ] Update `TOOLS[0]["description"]` in `app.py` to list the new document's topic areas
- [ ] Re-run `ingest.py` to rebuild the vector store
