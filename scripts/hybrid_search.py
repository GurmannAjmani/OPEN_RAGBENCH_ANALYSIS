import json
import os
import time
from collections import Counter
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


CHUNKS_FILE = Path("chunks/text_chunks.jsonl")
COLLECTION_NAME = "rag_chunks"
QUERY = "Can you explain the Precise quantification of bacterial strains after fecal microbiota transplantation delineates long-term engraftment and explains outcomes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DENSE_K = 30
BM25_K = 30
RERANK_K = 30
FINAL_K = 10
RRF_K = 60


def main():
    if not QUERY.strip():
        raise SystemExit("Set QUERY at the top of this script.")

    startup_start = time.perf_counter()
    records = {}
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records[record["chunk_id"]] = record
    load_time = time.perf_counter()

    chunk_list = list(records.values())
    bm25 = BM25Okapi([record["content"].lower().split() for record in chunk_list])
    bm25_index_time = time.perf_counter()

    load_dotenv()
    client = chromadb.PersistentClient(path=os.environ.get("CHROMA_PATH", "chroma_db"))
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    print(f"Ready in {time.perf_counter() - startup_start:.2f}s | {len(records):,} chunks")
    print(f"  Load chunks: {load_time - startup_start:.2f}s | BM25 index: {bm25_index_time - load_time:.2f}s")

    query = QUERY.strip()
    while query:
        total_start = time.perf_counter()
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_indices = sorted(range(len(chunk_list)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_K]
        bm25_ids = [chunk_list[i]["chunk_id"] for i in bm25_indices]
        bm25_time = time.perf_counter()

        query_embedding = embedder.encode(query, normalize_embeddings=True).tolist()
        dense = collection.query(query_embeddings=[query_embedding], n_results=DENSE_K, include=["documents"])
        dense_ids = dense["ids"][0]
        dense_time = time.perf_counter()

        rrf_scores = {}
        for rank, chunk_id in enumerate(dense_ids, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
        for rank, chunk_id in enumerate(bm25_ids, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
        candidate_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:RERANK_K]
        fusion_time = time.perf_counter()

        rerank_scores = dict(zip(
            candidate_ids,
            reranker.predict([(query, records[chunk_id]["content"]) for chunk_id in candidate_ids]),
        ))
        ranked_ids = sorted(rerank_scores, key=rerank_scores.get, reverse=True)[:FINAL_K]
        rerank_time = time.perf_counter()

        print(f"\nQuery: {query}\n")
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            record = records[chunk_id]
            preview = record["content"].replace("\n", " ")[:350]
            print(f"{rank}. {chunk_id}")
            print(f"   type={record['chunk_type']} parent={record['parent_id']}")
            print(f"   rrf={rrf_scores[chunk_id]:.4f} rerank={rerank_scores[chunk_id]:.4f}")
            print(f"   {preview}\n")

        types = Counter(records[chunk_id]["chunk_type"] for chunk_id in ranked_ids)
        text_count = sum(count for name, count in types.items() if name == "text")
        image_count = sum(count for name, count in types.items() if "image" in name)
        table_count = sum(count for name, count in types.items() if "table" in name)
        print(f"Retrieved: text={text_count} image={image_count} table={table_count}")
        print("Timing")
        print(f"  BM25:             {bm25_time - total_start:.2f}s")
        print(f"  Dense retrieval:  {dense_time - bm25_time:.2f}s")
        print(f"  RRF fusion:       {fusion_time - dense_time:.2f}s")
        print(f"  Cross-encoder:    {rerank_time - fusion_time:.2f}s")
        print(f"  Total:            {rerank_time - total_start:.2f}s")
        query = input("\nNext query (Enter to quit): ").strip()


if __name__ == "__main__":
    main()
