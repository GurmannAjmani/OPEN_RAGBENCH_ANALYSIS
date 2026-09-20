"""Chunk Open RAGBench text sections and ingest their embeddings into Chroma.
"""

import json
import os
from pathlib import Path

import psutil
import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


CORPUS_DIR = Path("open_ragbench/pdf/arxiv/corpus")
CHUNKS_FILE = Path("chunks/text_chunks.jsonl")
COLLECTION_NAME = "rag_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 1024
OVERLAP_TOKENS = 100


def main():
    load_dotenv()
    chroma_path = os.environ.get("CHROMA_PATH", "chroma_db")

    files = sorted(CORPUS_DIR.glob("*.json"))
    CHUNKS_FILE.parent.mkdir(exist_ok=True)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=MAX_TOKENS,
        chunk_overlap=OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    records = []

    for file_path in tqdm(files, desc="Chunking papers", unit="paper"):
        paper = json.loads(file_path.read_text(encoding="utf-8"))
        for section in paper.get("sections", []):
            section_id = section["section_id"]
            text = section.get("text") or ""
            for index, chunk_text in enumerate(splitter.split_text(text)):
                chunk_id = f"{paper['id']}:{section_id}:text:{index}"
                records.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper["id"],
                    "section_id": section_id,
                    "parent_id": f"{paper['id']}:{section_id}",
                    "chunk_type": "text",
                    "chunk_index": index,
                    "title": paper.get("title", ""),
                    "categories": paper.get("categories", []),
                    "content": chunk_text,
                })

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records):,} text chunk records to {CHUNKS_FILE}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [f"{r['title']}\n\n{r['content']}" for r in tqdm(records, desc="Embedding chunks", unit="chunk")],
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,

    )

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for start in tqdm(range(0, len(records), 500), desc="Ingesting chunks", unit="batch"):
        batch_records = records[start:start + 500]
        collection.upsert(
            ids=[record["chunk_id"] for record in batch_records],
            documents=[record["content"] for record in batch_records],
            embeddings=[embedding.tolist() for embedding in embeddings[start:start + 500]],
            metadatas=[
                {
                    "paper_id": record["paper_id"],
                    "section_id": record["section_id"],
                    "parent_id": record["parent_id"],
                    "chunk_type": record["chunk_type"],
                    "chunk_index": record["chunk_index"],
                    "metadata_json": json.dumps(record),
                }
                for record in batch_records
            ],
        )

    memory_mb = psutil.Process().memory_info().rss / 1024**2
    print(f"Done. Process memory in use: {memory_mb:.1f} MB")


if __name__ == "__main__":
    main()
