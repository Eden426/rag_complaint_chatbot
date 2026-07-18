from src.rag import RetrievedChunk, build_prompt, extractive_answer


def sample_chunk():
    return RetrievedChunk(
        "The transfer remained pending for seven days.",
        {"complaint_id": "42", "product_category": "Money Transfer"},
        0.81,
    )


def test_prompt_numbers_sources_and_forbids_invention():
    prompt = build_prompt("Why are transfers delayed?", [sample_chunk()])
    assert "[Source 1]" in prompt
    assert "Complaint 42" in prompt
    assert "never invent" in prompt


def test_extractive_fallback_cites_evidence():
    answer = extractive_answer([sample_chunk()])
    assert "pending for seven days" in answer
    assert "[Source 1]" in answer


def test_extractive_fallback_handles_no_results():
    assert "enough information" in extractive_answer([])

