"""Stratified sampling, text chunking, embedding, and FAISS indexing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def stratified_sample(
    frame: pd.DataFrame, sample_size: int = 12_000, random_state: int = 42
) -> pd.DataFrame:
    """Sample products proportionally while retaining every available category."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if "product_category" not in frame:
        raise ValueError("product_category column is required")
    if len(frame) <= sample_size:
        return frame.sample(frac=1, random_state=random_state).reset_index(drop=True)

    fractions = frame["product_category"].value_counts(normalize=True)
    allocations = (fractions * sample_size).astype(int).clip(lower=1)
    remainder = sample_size - int(allocations.sum())
    for category in fractions.index[: abs(remainder)]:
        allocations[category] += 1 if remainder > 0 else -1

    parts = []
    for category, count in allocations.items():
        group = frame[frame["product_category"] == category]
        parts.append(group.sample(n=min(int(count), len(group)), random_state=random_state))
    sampled = pd.concat(parts).sample(frac=1, random_state=random_state)
    return sampled.reset_index(drop=True)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split on word boundaries using character-size windows with overlap."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    text = " ".join(str(text).split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
        next_space = text.find(" ", start)
        if next_space != -1 and start > 0 and text[start - 1] != " ":
            start = next_space + 1
    return [chunk for chunk in chunks if chunk]


def create_chunk_records(
    frame: pd.DataFrame, chunk_size: int = 500, overlap: int = 50
) -> list[dict]:
    records: list[dict] = []
    for _, row in frame.iterrows():
        chunks = chunk_text(row["cleaned_narrative"], chunk_size, overlap)
        for index, text in enumerate(chunks):
            records.append(
                {
                    "text": text,
                    "complaint_id": str(row.get("Complaint ID", row.name)),
                    "product_category": str(row["product_category"]),
                    "product": str(row.get("Product", "")),
                    "issue": str(row.get("Issue", "")),
                    "sub_issue": str(row.get("Sub-issue", "")),
                    "company": str(row.get("Company", "")),
                    "state": str(row.get("State", "")),
                    "date_received": str(row.get("Date received", "")),
                    "chunk_index": index,
                    "total_chunks": len(chunks),
                }
            )
    return records


def build_index(
    records: list[dict], output_dir: Path, model_name: str = DEFAULT_MODEL
) -> None:
    if not records:
        raise ValueError("No text chunks were produced")
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [record["text"] for record in records],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings))

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "complaints.faiss"))
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "config.json").write_text(
        json.dumps({"model_name": model_name, "dimension": embeddings.shape[1], "chunks": len(records)}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/filtered_complaints.csv"))
    parser.add_argument("--output", type=Path, default=Path("vector_store"))
    parser.add_argument("--sample-size", type=int, default=12_000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    cleaned = pd.read_csv(args.input, low_memory=False)
    sample = stratified_sample(cleaned, args.sample_size)
    records = create_chunk_records(sample, args.chunk_size, args.overlap)
    build_index(records, args.output, args.model)
    print(f"Indexed {len(records):,} chunks from {len(sample):,} complaints")


if __name__ == "__main__":
    main()

