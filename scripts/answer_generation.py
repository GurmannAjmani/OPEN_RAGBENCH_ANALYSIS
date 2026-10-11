import json
import os
from pathlib import Path

import chromadb
import requests
import tiktoken
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


CHUNKS_FILE = Path("chunks/text_chunks.jsonl")
COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "qwen/qwen3.8-27b"
DENSE_K = 30
BM25_K = 30
RERANK_K = 30
FINAL_K = 8
RRF_K = 60


def groq_chat(messages, keys):
    last_error = None
    for key in keys:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.2, "max_completion_tokens": 700},
                timeout=90,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as error:
            last_error = error
    raise RuntimeError(f"All Groq API keys failed: {last_error}")


def main():
    load_dotenv()
    keys = [os.environ.get(f"GROQ_API_KEY_{i}") for i in range(1, 4)]
    keys = [key for key in keys if key]
    if not keys:
        raise SystemExit("Add GROQ_API_KEY_1, GROQ_API_KEY_2, or GROQ_API_KEY_3 to .env")

    records = {}
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records[record["chunk_id"]] = record

    chunk_list = list(records.values())
    bm25 = BM25Okapi([record["content"].lower().split() for record in chunk_list])
    client = chromadb.PersistentClient(path=os.environ.get("CHROMA_PATH", "chroma_db"))
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    summary = ""
    recent_turns = []

    print("Ready. Type exit to stop.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        bm25_scores = bm25.get_scores(question.lower().split())
        bm25_ids = [
            chunk_list[i]["chunk_id"]
            for i in sorted(range(len(chunk_list)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_K]
        ]
        embedding = embedder.encode(question, normalize_embeddings=True).tolist()
        dense_ids = collection.query(query_embeddings=[embedding], n_results=DENSE_K)["ids"][0]

        rrf_scores = {}
        for rank, chunk_id in enumerate(dense_ids, start=1):
            if chunk_id in records:
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
        for rank, chunk_id in enumerate(bm25_ids, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (RRF_K + rank)

        candidate_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:RERANK_K]
        rerank_scores = reranker.predict([(question, records[chunk_id]["content"]) for chunk_id in candidate_ids])
        chunk_ids = [
            chunk_id
            for chunk_id, _ in sorted(zip(candidate_ids, rerank_scores), key=lambda pair: pair[1], reverse=True)
        ][:FINAL_K]
        context = "\n\n".join(f"[{chunk_id}]\n{records[chunk_id]['content']}" for chunk_id in chunk_ids)
        print(f"Context tokens: {len(tokenizer.encode(context, disallowed_special=()))}")

        history = "\n".join(f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in recent_turns)
        answer = groq_chat([
            {"role": "system", "content": "Answer using only the retrieved context. If it is insufficient, say so. Cite chunk IDs in square brackets for factual claims."},
            {"role": "user", "content": f"Conversation summary:\n{summary or 'None'}\n\nRecent turns:\n{history or 'None'}\n\nRetrieved context:\n{context}\n\nQuestion: {question}"},
        ], keys)
        print(f"\nAssistant: {answer}")
        recent_turns.append({"question": question, "answer": answer})

        if len(recent_turns) == 3:
            summary = groq_chat([
                {"role": "system", "content": "Create a concise factual memory of this conversation for future turns. Preserve unresolved questions, constraints, conclusions, and cited identifiers."},
                {"role": "user", "content": f"Existing memory:\n{summary or 'None'}\n\nNew turns:\n{json.dumps(recent_turns, ensure_ascii=False)}"},
            ], keys)
            recent_turns = []
            print("Conversation memory updated.")


if __name__ == "__main__":
    main()
