from pathlib import Path
import json
from typing import List, Dict

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
VECTOR_DIR = Path("vector_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_whitespace(text: str) -> str:
    """
    Remove unnecessary spaces and newlines.
    """
    return " ".join(text.split())


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:

    text = normalize_whitespace(text)

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# READ PDF FILES
# ============================================================

def load_pdf_pages() -> List[Dict]:

    pdf_files = sorted(
        DATA_DIR.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            "No PDF files found inside the data folder."
        )

    records = []

    print("=" * 70)
    print("READING INDUSTRIAL DOCUMENTS")
    print("=" * 70)

    for pdf_path in pdf_files:

        print(
            f"\nReading: {pdf_path.name}"
        )

        reader = PdfReader(
            str(pdf_path)
        )

        print(
            f"Pages: {len(reader.pages)}"
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            text = normalize_whitespace(
                text
            )

            if not text:

                print(
                    f"  Page {page_number}: "
                    f"No text found"
                )

                continue

            chunks = chunk_text(
                text
            )

            print(
                f"  Page {page_number}: "
                f"{len(chunks)} chunks"
            )

            for chunk_id, chunk in enumerate(
                chunks,
                start=1
            ):

                records.append({

                    "source":
                        pdf_path.name,

                    "page":
                        page_number,

                    "chunk_id":
                        chunk_id,

                    "text":
                        chunk
                })

    return records


# ============================================================
# CREATE FAISS VECTOR DATABASE
# ============================================================

def build_faiss_index(
    records: List[Dict]
) -> None:

    texts = [
        record["text"]
        for record in records
    ]

    print("\n")
    print("=" * 70)
    print("LOADING EMBEDDING MODEL")
    print("=" * 70)

    print(
        f"Model: {EMBEDDING_MODEL}"
    )

    embedder = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("\n")
    print("=" * 70)
    print("GENERATING EMBEDDINGS")
    print("=" * 70)

    print(
        f"Number of chunks: {len(texts)}"
    )

    embeddings = embedder.encode(

        texts,

        batch_size=32,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    print(
        f"\nEmbedding dimension: {dimension}"
    )

    # Inner-product FAISS index.
    # Because embeddings are normalized,
    # inner product corresponds to cosine similarity.
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    # Create vector_db folder if necessary.
    VECTOR_DIR.mkdir(
        exist_ok=True
    )

    # Save FAISS index.
    index_path = (
        VECTOR_DIR /
        "index.faiss"
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    # Save metadata.
    metadata_path = (
        VECTOR_DIR /
        "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n")
    print("=" * 70)
    print("INDEX BUILT SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Total chunks: {len(records)}"
    )

    print(
        f"Embedding dimension: {dimension}"
    )

    print(
        f"FAISS index: {index_path}"
    )

    print(
        f"Metadata: {metadata_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("AI INDUSTRIAL CHATBOT")
    print("DOCUMENT INDEXING")
    print("=" * 70)

    records = load_pdf_pages()

    if not records:

        raise RuntimeError(
            "No text chunks were created from the PDFs."
        )

    print(
        f"\nTotal document chunks: "
        f"{len(records)}"
    )

    build_faiss_index(
        records
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()