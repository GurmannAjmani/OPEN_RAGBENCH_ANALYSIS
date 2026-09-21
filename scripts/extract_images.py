"""Decode base64 images from corpus to disk and extract surrounding text context."""

import base64
import csv
import json
import re
from pathlib import Path

import tiktoken
from tqdm.auto import tqdm


CORPUS_DIR = Path("open_ragbench/pdf/arxiv/corpus")
IMAGES_DIR = Path("assets/images")
JSON_OUT = IMAGES_DIR / "image_context.json"
CSV_OUT = IMAGES_DIR / "image_context.csv"
CONTEXT_TOKENS = 150  # tokens before and after the image placeholder

TOKENIZER = tiktoken.get_encoding("cl100k_base")

FIELDNAMES = ["image_id", "paper_id", "section_id", "image_path", "context"]


def tokens_around(text: str, placeholder: str, n: int) -> str:
    """Return up to n tokens before and n tokens after `placeholder` in `text`."""
    idx = text.find(placeholder)
    if idx == -1:
        # Placeholder not found — fall back to first/last n tokens of section text
        toks = TOKENIZER.encode(text)
        window = toks[:n * 2]
        return TOKENIZER.decode(window)

    before_text = text[:idx]
    after_text = text[idx + len(placeholder):]

    before_toks = TOKENIZER.encode(before_text)[-n:]
    after_toks = TOKENIZER.encode(after_text)[:n]

    return TOKENIZER.decode(before_toks) + placeholder + TOKENIZER.decode(after_toks)


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print("started")
    # Load already-processed image IDs to allow resuming
    processed = set()
    existing_rows = []
    if JSON_OUT.exists():
        try:
            existing_rows = json.loads(JSON_OUT.read_text(encoding="utf-8"))
            processed = {r["image_id"] for r in existing_rows}
            print(f"Resuming: {len(processed)} images already processed.")
        except Exception:
            pass
    print("ended.opening csv")

    # Open CSV in append mode; write header only if file is new/empty
    csv_is_new = not CSV_OUT.exists() or CSV_OUT.stat().st_size == 0
    csv_file = CSV_OUT.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if csv_is_new:
        writer.writeheader()
        csv_file.flush()
    print("collecting all images")
    # Collect all images across corpus
    files = sorted(CORPUS_DIR.glob("*.json"))
    all_images = []
    for file_path in files:
        paper = json.loads(file_path.read_text(encoding="utf-8"))
        paper_id = paper["id"]
        for section in paper.get("sections", []):
            section_id = section["section_id"]
            section_text = section.get("text", "")
            for image_id, image_b64 in (section.get("images") or {}).items():
                all_images.append({
                    "image_id": f"{paper_id}:{section_id}:{image_id}",
                    "paper_id": paper_id,
                    "section_id": section_id,
                    "image_raw_id": image_id,
                    "image_b64": image_b64,
                    "section_text": section_text,
                })
    print("process started")
    for item in tqdm(all_images, desc="Extracting images", unit="image"):
        image_id = item["image_id"]
        if image_id in processed:
            continue

        # Decode base64 (strip data-URI prefix if present)
        b64_val = item["image_b64"]
        if b64_val.startswith("data:"):
            header, b64_val = b64_val.split(",", 1)
            # Determine extension from MIME type
            mime = header.split(";")[0].split(":")[1]  # e.g. image/jpeg
            ext = mime.split("/")[1]  # jpeg, png, gif …
            ext = "jpg" if ext == "jpeg" else ext
        else:
            ext = "png"

        image_path = (
            IMAGES_DIR
            / item["paper_id"]
            / str(item["section_id"])
            / Path(item["image_raw_id"]).stem
        ).with_suffix(f".{ext}")
        image_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already written to disk
        if not image_path.exists():
            image_path.write_bytes(base64.b64decode(b64_val))

        # Build context window around the image placeholder (e.g. "img-0.jpeg")
        placeholder = item["image_raw_id"]
        context = tokens_around(item["section_text"], placeholder, CONTEXT_TOKENS)

        row = {
            "image_id": image_id,
            "paper_id": item["paper_id"],
            "section_id": item["section_id"],
            "image_path": str(image_path),
            "context": context,
        }

        existing_rows.append(row)
        processed.add(image_id)

        # Append to CSV immediately
        writer.writerow(row)
        csv_file.flush()

        # Rewrite JSON incrementally
        JSON_OUT.write_text(
            json.dumps(existing_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    csv_file.close()
    print(f"\nDone. {len(existing_rows):,} images saved to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
