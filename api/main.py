import json
import os
from functools import lru_cache
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


CHUNKS_FILE = Path("chunks/text_chunks.jsonl")
ASSETS_DIR = Path("assets")
COLLECTION_NAME = "rag_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "qwen/qwen3.8-27b"
DENSE_K = 30
BM25_K = 30
RERANK_K = 30
FINAL_K = 8
RRF_K = 60
MAX_CONTEXT_CHARS = 18000

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


@lru_cache
def resources():
    records = {}
    with CHUNKS_FILE.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            records[record["chunk_id"]] = record

    chunks = list(records.values())
    return {
        "records": records,
        "chunks": chunks,
        "bm25": BM25Okapi([chunk["content"].lower().split() for chunk in chunks]),
        "collection": chromadb.PersistentClient(
            path=os.environ.get("CHROMA_PATH", "chroma_db")
        ).get_collection(COLLECTION_NAME),
        "embedder": SentenceTransformer(MODEL_NAME),
        "reranker": CrossEncoder(RERANKER_NAME),
    }


def retrieve(question):
    data = resources()
    records = data["records"]
    bm25_scores = data["bm25"].get_scores(question.lower().split())
    bm25_ids = [
        data["chunks"][index]["chunk_id"]
        for index in sorted(range(len(data["chunks"])), key=lambda index: bm25_scores[index], reverse=True)[:BM25_K]
    ]
    embedding = data["embedder"].encode(question, normalize_embeddings=True).tolist()
    dense_ids = data["collection"].query(query_embeddings=[embedding], n_results=DENSE_K)["ids"][0]

    scores = {}
    for rank, chunk_id in enumerate(dense_ids, start=1):
        if chunk_id in records:
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
    for rank, chunk_id in enumerate(bm25_ids, start=1):
        if chunk_id in records:
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)

    candidate_ids = sorted(scores, key=scores.get, reverse=True)[:RERANK_K]
    rerank_scores = data["reranker"].predict(
        [(question, records[chunk_id]["content"]) for chunk_id in candidate_ids]
    )
    return [
        records[chunk_id]
        for chunk_id, _ in sorted(zip(candidate_ids, rerank_scores), key=lambda item: item[1], reverse=True)[:FINAL_K]
    ]


def groq_stream(messages):
    errors = []
    for index in range(1, 4):
        key = os.environ.get(f"GROQ_API_KEY_{index}")
        if not key:
            continue
        try:
            stream = Groq(api_key=key).chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.2,
                max_completion_tokens=700,
                stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content
                if text:
                    yield text
            return
        except Exception as error:
            errors.append(f"Key {index}: {error}")
    raise RuntimeError("All Groq API keys failed. " + " | ".join(errors))


def event(kind, data):
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/chat")
def chat(question: str = Query(min_length=1)):
    def stream():
        try:
            chunks = retrieve(question)
            context_chunks = []
            context_size = 0
            for chunk in chunks:
                chunk_size = len(chunk["content"])
                if context_chunks and context_size + chunk_size > MAX_CONTEXT_CHARS:
                    break
                context_chunks.append(chunk)
                context_size += chunk_size

            images = [
                {
                    "chunk_id": chunk["chunk_id"],
                    "path": "http://127.0.0.1:8001/assets/" + chunk["image_path"].replace("\\", "/").removeprefix("assets/"),
                }
                for chunk in context_chunks
                if chunk["chunk_type"] == "image_context" and chunk.get("image_path")
            ]
            yield event("context", {"chunks": context_chunks, "images": images})

            context = "\n\n".join(f"[{chunk['chunk_id']}]\n{chunk['content']}" for chunk in context_chunks)
            messages = [
                {"role": "system", "content": "Answer using only the retrieved context. Cite chunk IDs in square brackets for factual claims. Do not include Markdown image links; relevant retrieved figures are displayed below the answer."},
                {"role": "user", "content": f"Retrieved context:\n{context}\n\nQuestion: {question}"},
            ]
            for text in groq_stream(messages):
                yield event("token", text)
            yield event("done", {})
        except Exception as error:
            yield event("error", str(error))

    return StreamingResponse(stream(), media_type="text/event-stream")
