import json
import os
import sys
from pathlib import Path
import chromadb

k = int(sys.argv[1]) if len(sys.argv) > 1 else 5

print(f"=== First {k} Chunks from text_chunks.jsonl ===")
with Path("chunks/text_chunks.jsonl").open(encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= k:
            break
        print(f"\n[Chunk {i+1}]")
        print(json.dumps(json.loads(line), indent=2, ensure_ascii=False))

print(f"\n=== First {k} Records from Chroma DB (rag_chunks) ===")
client = chromadb.PersistentClient(path=os.environ.get("CHROMA_PATH", "chroma_db"))
collection = client.get_collection("rag_chunks")
results = collection.get(limit=k, include=["metadatas", "documents"])

for i in range(len(results["ids"])):
    print(f"\n[Chroma Record {i+1}]")
    print(f"ID: {results['ids'][i]}")
    print(f"Metadata: {json.dumps(results['metadatas'][i], indent=2, ensure_ascii=False)}")
    print(f"Document: {results['documents'][i][:300]}...")
