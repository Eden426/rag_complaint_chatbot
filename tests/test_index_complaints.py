import pandas as pd
import pytest

from src.index_complaints import chunk_text, create_chunk_records, stratified_sample


def test_chunk_text_overlaps_without_exceeding_target():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunk_text_rejects_invalid_window():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=50, overlap=50)


def test_stratified_sample_preserves_categories():
    frame = pd.DataFrame(
        {
            "product_category": ["Credit Card"] * 80 + ["Personal Loan"] * 20,
            "cleaned_narrative": ["text"] * 100,
        }
    )
    sampled = stratified_sample(frame, sample_size=20)
    assert sampled["product_category"].value_counts().to_dict() == {"Credit Card": 16, "Personal Loan": 4}


def test_chunk_metadata_is_traceable():
    frame = pd.DataFrame(
        [{"Complaint ID": 99, "Product": "Credit card", "product_category": "Credit Card", "cleaned_narrative": "card was charged twice"}]
    )
    records = create_chunk_records(frame)
    assert records[0]["complaint_id"] == "99"
    assert records[0]["product_category"] == "Credit Card"
    assert records[0]["total_chunks"] == 1

