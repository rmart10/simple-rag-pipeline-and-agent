"""
Ingest a PDF into a Chroma vector database.
Run once before starting the app:  python ingest.py
"""

import pypdf
import chromadb

PDF_PATH = "documents/114-6133-000.pdf"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "manual"
CHUNK_SIZE = 300   # words per chunk
CHUNK_OVERLAP = 50  # words of overlap between chunks


def load_pdf(path: str) -> list[dict]:
    """Return a list of {page, text} dicts, one per PDF page."""
    reader = pypdf.PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def make_chunks(pages: list[dict]) -> list[dict]:
    """Split the full document into overlapping word-based chunks.

    Words are flattened into a single (word, page_number) list before
    windowing, so the sliding window can cross page boundaries freely.
    Each chunk is tagged with the page where its first word appears.
    """
    # Flatten every page into one ordered list of (word, page_number) pairs
    word_pages: list[tuple[str, int]] = []
    for page in pages:
        for word in page["text"].split():
            word_pages.append((word, page["page"]))

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(word_pages), step):
        items = word_pages[start : start + CHUNK_SIZE]
        if len(items) < 20:  # skip tiny trailing fragments
            continue
        chunks.append(
            {
                "text": " ".join(w for w, _ in items),
                "page": items[0][1],  # page where this chunk begins
            }
        )
    return chunks


def main():
    print(f"Loading {PDF_PATH} …")
    pages = load_pdf(PDF_PATH)
    print(f"  {len(pages)} pages extracted")

    chunks = make_chunks(pages)
    print(f"  {len(chunks)} chunks created")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Wipe and recreate the collection so re-running is safe
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

    print(f"  Done — {len(chunks)} chunks stored in '{CHROMA_PATH}'")


if __name__ == "__main__":
    main()
