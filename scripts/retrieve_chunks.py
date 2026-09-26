import json
import sys
from collections import Counter
from pathlib import Path

PARENT_ID = "2401.01872v2:0"

CHUNKS_FILE = Path("chunks/text_chunks.jsonl")
RESULT_DIR = Path("result")


def main():
    if not CHUNKS_FILE.exists():
        print(f"Error: {CHUNKS_FILE} not found.")
        sys.exit(1)

    parent_id = PARENT_ID.strip() if PARENT_ID and PARENT_ID.strip() else None
    if not parent_id:
        with CHUNKS_FILE.open(encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                if "parent_id" in chunk:
                    parent_id = chunk["parent_id"]
                    break

    if not parent_id:
        print("Error: Could not determine a parent_id.")
        sys.exit(1)

    print(f"\n=== Retrieving all chunks for parent_id: '{parent_id}' ===")

    # Retrieve all chunks matching parent_id across all content types (text, image, table)
    matching_chunks = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk.get("parent_id") == parent_id:
                matching_chunks.append(chunk)

    print(f"Found {len(matching_chunks)} chunks for parent_id '{parent_id}'.")

    if not matching_chunks:
        print("No matching chunks found.")
        return

    # Display chunk type summary
    type_counts = Counter(c.get("chunk_type", "unknown") for c in matching_chunks)
    print("\nChunk Type Breakdown:")
    for chunk_type, count in type_counts.items():
        print(f"  - {chunk_type}: {count}")

    # Display preview of retrieved chunks
    for i, chunk in enumerate(matching_chunks, 1):
        print(f"\n--- [Chunk {i}/{len(matching_chunks)}] ({chunk.get('chunk_type')}) ---")
        print(f"ID      : {chunk.get('chunk_id')}")
        content = chunk.get("content", "")
        preview = content[:200] + ("..." if len(content) > 200 else "")
        print(f"Content : {preview}")

    # Create result directory and save retrieved chunks
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    safe_filename = parent_id.replace(":", "_") + "_chunks.json"
    output_file = RESULT_DIR / safe_filename

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(matching_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(matching_chunks)} chunks to: {output_file}")


if __name__ == "__main__":
    main()


