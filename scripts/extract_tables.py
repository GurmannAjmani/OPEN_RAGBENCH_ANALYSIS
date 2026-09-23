"""Extract tables from corpus to disk and extract surrounding text context + table header."""

import csv
import json
from pathlib import Path

import tiktoken
from tqdm.auto import tqdm

CORPUS_DIR = Path("open_ragbench/pdf/arxiv/corpus")
TABLES_DIR = Path("assets/tables")
JSON_OUT = TABLES_DIR / "table_context.json"
CSV_OUT = TABLES_DIR / "table_context.csv"
CONTEXT_TOKENS = 50

TOKENIZER = tiktoken.get_encoding("cl100k_base")
FIELDNAMES = ["table_id", "paper_id", "section_id", "table_path", "context"]


def get_table_context(text: str, placeholder: str, table_str: str, n: int) -> str:
    lines = [l for l in table_str.strip().split("\n") if l.strip()]
    header = lines[0] if lines else ""

    idx = text.find(placeholder)
    if idx == -1:
        raw_id = placeholder.split("[")[1].split("]")[0] if "[" in placeholder else placeholder
        idx = text.find(raw_id)

    if idx == -1:
        toks = TOKENIZER.encode(text)
        before = TOKENIZER.decode(toks[:n])
        after = TOKENIZER.decode(toks[-n:])
    else:
        before_text = text[:idx]
        after_text = text[idx + len(placeholder):]
        before = TOKENIZER.decode(TOKENIZER.encode(before_text)[-n:])
        after = TOKENIZER.decode(TOKENIZER.encode(after_text)[:n])

    return f"{before}\n{header}\n{after}".strip()


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    processed = set()
    existing_rows = []
    if JSON_OUT.exists():
        try:
            existing_rows = json.loads(JSON_OUT.read_text(encoding="utf-8"))
            processed = {r["table_id"] for r in existing_rows}
            print(f"Resuming: {len(processed)} tables already processed.")
        except Exception:
            pass

    csv_is_new = not CSV_OUT.exists() or CSV_OUT.stat().st_size == 0
    csv_file = CSV_OUT.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if csv_is_new:
        writer.writeheader()
        csv_file.flush()

    files = sorted(CORPUS_DIR.glob("*.json"))
    all_tables = []
    for file_path in files:
        paper = json.loads(file_path.read_text(encoding="utf-8"))
        paper_id = paper["id"]
        for section in paper.get("sections", []):
            section_id = section["section_id"]
            section_text = section.get("text", "")
            for table_raw_id, table_str in (section.get("tables") or {}).items():
                all_tables.append({
                    "table_id": f"{paper_id}:{section_id}:{table_raw_id}",
                    "paper_id": paper_id,
                    "section_id": section_id,
                    "table_raw_id": table_raw_id,
                    "table_str": table_str,
                    "section_text": section_text,
                })

    for item in tqdm(all_tables, desc="Extracting tables", unit="table"):
        table_id = item["table_id"]
        if table_id in processed:
            continue

        table_path = (
            TABLES_DIR
            / item["paper_id"]
            / str(item["section_id"])
            / f"{item['table_raw_id']}.md"
        )
        table_path.parent.mkdir(parents=True, exist_ok=True)

        if not table_path.exists():
            table_path.write_text(item["table_str"], encoding="utf-8")

        placeholder = f"![{item['table_raw_id']}]({item['table_raw_id']})"
        context = get_table_context(
            item["section_text"], placeholder, item["table_str"], CONTEXT_TOKENS
        )

        row = {
            "table_id": table_id,
            "paper_id": item["paper_id"],
            "section_id": item["section_id"],
            "table_path": str(table_path),
            "context": context,
        }

        existing_rows.append(row)
        processed.add(table_id)

        writer.writerow(row)
        csv_file.flush()

        JSON_OUT.write_text(
            json.dumps(existing_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    csv_file.close()
    print(f"\nDone. {len(existing_rows):,} tables saved to {TABLES_DIR}")


if __name__ == "__main__":
    main()
