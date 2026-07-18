"""Retrieval-augmented complaint analysis with cited source excerpts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

SYSTEM_PROMPT = """You are a careful financial complaint analyst for CrediTrust.
Answer the question using only the complaint excerpts below. Synthesize recurring
themes, mention meaningful differences between products, and cite evidence using
[Source N]. Do not treat the excerpts as a statistically representative sample.
If the excerpts do not support an answer, say that there is not enough information.
Keep the answer concise and never invent facts, counts, or causes."""


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict
    score: float

    @property
    def citation(self) -> str:
        complaint_id = self.metadata.get("complaint_id", "unknown")
        product = self.metadata.get("product_category", "unknown product")
        return f"Complaint {complaint_id} · {product}"


class ComplaintRetriever:
    """Load and search a persisted FAISS complaint index."""

    def __init__(self, store_dir: Path | str = "vector_store") -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self.store_dir = Path(store_dir)
        config = json.loads((self.store_dir / "config.json").read_text(encoding="utf-8"))
        self.model = SentenceTransformer(config["model_name"])
        self.index = faiss.read_index(str(self.store_dir / "complaints.faiss"))
        with (self.store_dir / "metadata.jsonl").open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if self.index.ntotal != len(self.records):
            raise ValueError("FAISS index and metadata have different record counts")

    def search(
        self, question: str, top_k: int = 5, product: str | None = None
    ) -> list[RetrievedChunk]:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = self.model.encode([question], normalize_embeddings=True).astype("float32")
        candidate_count = min(len(self.records), max(top_k * 20, top_k))
        scores, indices = self.index.search(np.ascontiguousarray(query), candidate_count)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            record = self.records[int(index)]
            if product and record.get("product_category") != product:
                continue
            results.append(RetrievedChunk(record["text"], record, float(score)))
            if len(results) == top_k:
                break
        return results


def build_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[Source {number}] {chunk.citation}\n{chunk.text}"
        for number, chunk in enumerate(chunks, start=1)
    )
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context or '(no relevant excerpts)'}\n\nQuestion: {question}\n\nAnswer:"


def extractive_answer(chunks: Sequence[RetrievedChunk]) -> str:
    """Safe fallback when no hosted LLM key is configured."""
    if not chunks:
        return "I don't have enough information in the retrieved complaints to answer that question."
    lines = ["The retrieved complaints highlight the following evidence:"]
    lines.extend(f"- {chunk.text} [Source {i}]" for i, chunk in enumerate(chunks[:3], 1))
    lines.append("This is an extractive summary; configure an LLM API key for cross-source synthesis.")
    return "\n".join(lines)


def generate_answer(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Generate a grounded answer, falling back to cited extraction offline."""
    if not os.getenv("OPENAI_API_KEY"):
        return extractive_answer(chunks)
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=build_prompt(question, chunks),
        temperature=0.1,
    )
    return response.output_text.strip()


def answer_question(
    question: str,
    retriever: ComplaintRetriever,
    top_k: int = 5,
    product: str | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    chunks = retriever.search(question, top_k=top_k, product=product)
    return generate_answer(question, chunks), chunks

