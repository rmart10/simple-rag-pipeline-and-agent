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
    """Split each page's text into overlapping word-based chunks."""
    chunks = []
    for page in pages:
        words = page["text"].split()
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for start in range(0, len(words), step):
            chunk_words = words[start : start + CHUNK_SIZE]
            if len(chunk_words) < 20:  # skip tiny trailing fragments
                continue
            chunks.append(
                {
                    "text": " ".join(chunk_words),
                    "page": page["page"],
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
