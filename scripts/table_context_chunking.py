import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

TABLE_CONTEXT_JSON = Path("assets/tables/table_context.json")
TEXT_CHUNKS_JSONL  = Path("chunks/text_chunks.jsonl")
MODEL_NAME         = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME    = "rag_chunks"

load_dotenv()
table_rows = json.loads(TABLE_CONTEXT_JSON.read_text(encoding="utf-8"))

# Build parent_id -> {title, categories} from existing text chunks
paper_meta = {}
existing_ids = set()
with TEXT_CHUNKS_JSONL.open(encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        pid = chunk.get("parent_id")
        if pid and pid not in paper_meta:
            paper_meta[pid] = {"title": chunk.get("title", ""), "categories": chunk.get("categories", [])}
        existing_ids.add(chunk["chunk_id"])

print(f"{len(existing_ids):,} existing chunks | {len(table_rows):,} table context records")

# Build new chunks, skipping already processed ones
new_chunks = []
with TEXT_CHUNKS_JSONL.open("a", encoding="utf-8") as out:
    for row in tqdm(table_rows, desc="Building table context chunks", unit="chunk"):
        chunk_id  = f"{row['table_id']}:context"
        parent_id = f"{row['paper_id']}:{row['section_id']}"

        if chunk_id in existing_ids:
            continue

        meta = paper_meta.get(parent_id, {"title": "", "categories": []})
        chunk = {
            "chunk_id":   chunk_id,
            "paper_id":   row["paper_id"],
            "section_id": row["section_id"],
            "parent_id":  parent_id,
            "chunk_type": "table_context",
            "table_id":   row["table_id"],
            "table_path": row["table_path"],
            "title":      meta["title"],
            "categories": meta["categories"],
            "content":    row["context"],
        }
        out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        existing_ids.add(chunk_id)
        new_chunks.append(chunk)

print(f"Appended {len(new_chunks):,} new table context chunks to {TEXT_CHUNKS_JSONL}")

if not new_chunks:
    print("Nothing new to embed.")
    exit(0)

# Embed using the same model as chunk_and_ingest.py
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    [f"{c['title']}\n\n{c['content']}" for c in new_chunks],
    show_progress_bar=True,
    batch_size=64,
    normalize_embeddings=True,
)

# Upsert into the same Chroma collection
client = chromadb.PersistentClient(path=os.environ.get("CHROMA_PATH", "chroma_db"))
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

for start in tqdm(range(0, len(new_chunks), 500), desc="Ingesting into Chroma", unit="batch"):
    batch = new_chunks[start:start + 500]
    collection.upsert(
        ids=[c["chunk_id"] for c in batch],
        documents=[c["content"] for c in batch],
        embeddings=[e.tolist() for e in embeddings[start:start + 500]],
        metadatas=[
            {
                "paper_id":   c["paper_id"],
                "section_id": c["section_id"],
                "parent_id":  c["parent_id"],
                "chunk_type": c["chunk_type"],
                "table_id":   c["table_id"],
                "table_path": c["table_path"],
                "metadata_json": json.dumps(c),
            }
            for c in batch
        ],
    )

print("Done.")
