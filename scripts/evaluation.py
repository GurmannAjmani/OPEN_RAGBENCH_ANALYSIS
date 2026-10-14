import csv
import json
from pathlib import Path

from tqdm.auto import tqdm

from api.main import retrieve


DATASET_DIR = Path("open_ragbench/pdf/arxiv")
OUTPUT_FILE = Path("result/retrieval_evaluation.csv")
FIELDS = [
    "query_id",
    "query",
    "source",
    "type",
    "gold_doc_id",
    "gold_section_id",
    "retrieved_chunk_ids",
    "retrieved_chunk_types",
    "gold_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_8",
    "mrr_at_8",
    "status",
    "error",
]


def main():
    queries = json.loads((DATASET_DIR / "queries.json").read_text(encoding="utf-8"))
    qrels = json.loads((DATASET_DIR / "qrels.json").read_text(encoding="utf-8"))
    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    completed = set()
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(encoding="utf-8", newline="") as file:
            completed = {
                row["query_id"]
                for row in csv.DictReader(file)
                if row["status"] == "done"
            }

    write_header = not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0
    with OUTPUT_FILE.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        for query_id, item in tqdm(queries.items(), desc="Evaluating retrieval", unit="query"):
            if query_id in completed:
                continue

            gold = qrels[query_id]
            row = {
                "query_id": query_id,
                "query": item["query"],
                "source": item["source"],
                "type": item["type"],
                "gold_doc_id": gold["doc_id"],
                "gold_section_id": gold["section_id"],
                "status": "done",
                "error": "",
            }

            try:
                chunks = retrieve(item["query"])
                rank = next((
                    index
                    for index, chunk in enumerate(chunks, start=1)
                    if chunk["paper_id"] == gold["doc_id"] and chunk["section_id"] == gold["section_id"]
                ), 0)
                row.update({
                    "retrieved_chunk_ids": json.dumps([chunk["chunk_id"] for chunk in chunks]),
                    "retrieved_chunk_types": json.dumps([chunk["chunk_type"] for chunk in chunks]),
                    "gold_rank": rank,
                    "hit_at_1": int(rank == 1),
                    "hit_at_3": int(0 < rank <= 3),
                    "hit_at_5": int(0 < rank <= 5),
                    "hit_at_8": int(0 < rank <= 8),
                    "mrr_at_8": round(1 / rank, 6) if rank else 0,
                })
            except Exception as error:
                row.update({
                    "retrieved_chunk_ids": "[]",
                    "retrieved_chunk_types": "[]",
                    "gold_rank": 0,
                    "hit_at_1": 0,
                    "hit_at_3": 0,
                    "hit_at_5": 0,
                    "hit_at_8": 0,
                    "mrr_at_8": 0,
                    "status": "error",
                    "error": str(error),
                })

            writer.writerow(row)
            file.flush()

    with OUTPUT_FILE.open(encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["status"] == "done"]

    for name, group in [("all", rows)] + [
        (source, [row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    ]:
        if group:
            print(
                f"{name}: {len(group):,} queries | "
                f"Hit@1 {sum(int(row['hit_at_1']) for row in group) / len(group):.3f} | "
                f"Hit@3 {sum(int(row['hit_at_3']) for row in group) / len(group):.3f} | "
                f"Hit@5 {sum(int(row['hit_at_5']) for row in group) / len(group):.3f} | "
                f"Hit@8 {sum(int(row['hit_at_8']) for row in group) / len(group):.3f} | "
                f"MRR@8 {sum(float(row['mrr_at_8']) for row in group) / len(group):.3f}"
            )


if __name__ == "__main__":
    main()
