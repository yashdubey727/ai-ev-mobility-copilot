from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
KB_PATH = BASE_DIR / "rag" / "knowledge.md"
DB_PATH = BASE_DIR / "rag" / "chroma_db"


def chunk_text(text: str, chunk_size_words: int = 220, overlap_words: int = 60):
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size_words]
        chunk_str = " ".join(chunk).strip()
        if chunk_str:
            chunks.append(chunk_str)
        i += max(1, chunk_size_words - overlap_words)
    return chunks


def main():
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge file not found: {KB_PATH}")

    kb = KB_PATH.read_text(encoding="utf-8").strip()
    if not kb:
        raise ValueError(f"Knowledge file is empty: {KB_PATH}")

    chunks = chunk_text(kb)
    if not chunks:
        raise ValueError("Chunking produced 0 chunks. Check knowledge.md formatting/content.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).tolist()

    if not embeddings or len(embeddings) != len(chunks):
        raise ValueError(
            f"Embeddings missing or mismatch. chunks={len(chunks)} embeddings={len(embeddings) if embeddings else 0}"
        )

    client = chromadb.PersistentClient(path=str(DB_PATH))
    col = client.get_or_create_collection("ev_kb")

    # Reset for deterministic rebuild
    existing = col.get()
    if existing and existing.get("ids"):
        col.delete(ids=existing["ids"])

    ids = [f"kb_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "knowledge.md", "chunk": i} for i in range(len(chunks))]

    col.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

    print(f"OK: embedded {len(chunks)} chunks into {DB_PATH}")
    print("Sample chunk:")
    print(chunks[0][:200])


if __name__ == "__main__":
    main()