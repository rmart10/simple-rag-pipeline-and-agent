"""
RAG agent — Chamberlain garage door opener manual.
Requires:  ANTHROPIC_API_KEY in .env (or environment) + ingest.py already run.
Start with:  python app.py 
"""

import os
from dotenv import load_dotenv
import chromadb
import anthropic
import gradio as gr

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "manual"
MODEL = "claude-sonnet-4-6"
N_RESULTS = 4  # chunks to retrieve per query

# ── Clients ──────────────────────────────────────────────────────────────────

chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_collection(COLLECTION_NAME)
claude = anthropic.Anthropic()

# ── Tool definition ───────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_manual",
        "description": (
            "Search the Chamberlain garage door opener manual for information "
            "about installation, programming, the myQ app, safety, operation, "
            "adjustments, maintenance, troubleshooting, or repair parts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A short, focused search query.",
                }
            },
            "required": ["query"],
        },
    }
]

SYSTEM = (
    "You are a knowledgeable assistant for the Chamberlain garage door opener "
    "manual (models D1000, C1000, C3000, B3010, B3000, B5330, B4310, B2310, "
    "C4310). Always call search_manual before answering so your response is "
    "grounded in the manual. Cite the page number when relevant. If the manual "
    "does not contain the answer, say so clearly."
)


# ── RAG retrieval ─────────────────────────────────────────────────────────────

def search_manual(query: str) -> str:
    results = collection.query(query_texts=[query], n_results=N_RESULTS)
    sections = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        sections.append(f"[Page {meta['page']}]\n{doc}")
    return "\n\n---\n\n".join(sections)


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(user_message: str, history: list) -> str:
    """Run a ReAct-style agent loop and return the final text answer."""
    # Convert Gradio history (list of {"role", "content"} dicts or (user, assistant) tuples)
    # into the Claude messages format.
    messages = []
    for item in history:
        if isinstance(item, dict):
            # Gradio 6 may include extra fields (e.g. metadata) — keep only what Claude accepts
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            human, assistant = item
            if human:
                messages.append({"role": "user", "content": human})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_message})

    while True:
        response = claude.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return next(
                (b.text for b in response.content if hasattr(b, "text")),
                "Sorry, I couldn't generate a response.",
            )

        if response.stop_reason == "tool_use":
            # Execute every tool call the model requested
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = search_manual(block.input["query"])
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            return "Unexpected stop reason — please try again."


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def chat(message: str, history: list) -> str:
    return run_agent(message, history)


demo = gr.ChatInterface(
    fn=chat,
    title="Garage Door Opener Manual Assistant",
    description=(
        "Ask anything about your Chamberlain garage door opener. "
        "Answers are retrieved directly from the official manual."
    ),
    examples=[
        "How do I program a new remote control?",
        "Why won't my door close?",
        "How do I connect the myQ app to Wi-Fi?",
        "How do I adjust the travel limits?",
        "What maintenance does the opener need?",
    ],
)

if __name__ == "__main__":
    demo.launch()
