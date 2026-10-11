import json
import os
import sys
import types
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ModuleNotFoundError:
    vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
    vertexai.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[vertexai.__name__] = vertexai

from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm


DATASET_DIR = Path("open_ragbench/pdf/arxiv")
CHUNKS_FILE = Path("chunks/text_chunks.jsonl")
OUTPUT_FILE = Path("result/ragas_text.jsonl")
COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "qwen3-vl:4b"
DENSE_K = 30
BM25_K = 30
RERANK_K = 30
FINAL_K = 8
RRF_K = 60


def ollama_answer(question, contexts, client):
    prompt = "\n\n".join(contexts)
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=0.3,
        max_completion_tokens=700,
        messages=[
            {"role": "system", "content": "Answer only from the retrieved context. If it is insufficient, say so."},
            {"role": "user", "content": f"Retrieved context:\n{prompt}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()


def ragas_scores(question, contexts, answer, reference, client):
    dataset = EvaluationDataset.from_list([{
        "user_input": question,
        "retrieved_contexts": contexts,
        "response": answer,
        "reference": reference,
    }])
    evaluator = llm_factory(OLLAMA_MODEL, provider="openai", client=client)
    result = evaluate(
        dataset=dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator,
        raise_exceptions=True,
    )
    return {
        name: float(value) if hasattr(value, "item") else value
        for name, value in result.to_pandas().iloc[0].to_dict().items()
    }


def main():
    load_dotenv()
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    ollama = OpenAI(api_key="ollama", base_url=f"{ollama_url}/v1")

    queries = json.loads((DATASET_DIR / "queries.json").read_text(encoding="utf-8"))
    qrels = json.loads((DATASET_DIR / "qrels.json").read_text(encoding="utf-8"))
    answers = json.loads((DATASET_DIR / "answers.json").read_text(encoding="utf-8"))
    test_set = [(query_id, item) for query_id, item in queries.items() if item["source"] == "text"]

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

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    completed = set()
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(encoding="utf-8") as f:
            completed = {
                row["query_id"]
                for line in f if line.strip()
                for row in [json.loads(line)]
                if "ragas" in row
            }

    with OUTPUT_FILE.open("a", encoding="utf-8") as output:
        for query_id, item in tqdm(test_set, desc="RAGAS text evaluation", unit="query"):
            if query_id in completed:
                continue
            question = item["query"]
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
            contexts = [records[chunk_id]["content"] for chunk_id in chunk_ids]
            gold = qrels[query_id]
            hit_at_8 = any(
                records[chunk_id]["paper_id"] == gold["doc_id"] and records[chunk_id]["section_id"] == gold["section_id"]
                for chunk_id in chunk_ids
            )

            try:
                answer = ollama_answer(question, contexts, ollama)
                scores = ragas_scores(question, contexts, answer, answers[query_id], ollama)
                row = {
                    "query_id": query_id,
                    "query": question,
                    "type": item["type"],
                    "source": item["source"],
                    "reference": answers[query_id],
                    "answer": answer,
                    "retrieved_chunk_ids": chunk_ids,
                    "retrieved_contexts": contexts,
                    "gold_doc_id": gold["doc_id"],
                    "gold_section_id": gold["section_id"],
                    "hit_at_8": hit_at_8,
                    "ragas": scores,
                }
            except Exception as error:
                tqdm.write(f"Query {query_id} failed: {error}")
                row = {"query_id": query_id, "query": question, "error": str(error)}

            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
