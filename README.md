# Intelligent Complaint Analysis for Financial Services

CrediTrust's internal complaint analyst turns CFPB consumer narratives into cited,
searchable evidence. It includes reproducible preprocessing and EDA, stratified
sampling, MiniLM embeddings in FAISS, a grounded RAG pipeline, and a Streamlit chat
interface with product filters and inspectable sources.

## Architecture

```text
CFPB CSV → filter/clean + EDA → stratified sample → overlapping chunks
         → MiniLM embeddings → FAISS index → retrieve top-k excerpts
         → grounded prompt → LLM or offline cited fallback → Streamlit UI
```

## Quick start

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.preprocess --input data/raw/complaints.csv
python -m src.index_complaints
streamlit run app.py
```

Place the CFPB export in `data/raw/complaints.csv`. Generated CSVs, plots, and vector
indexes are intentionally ignored by Git because they are large and reproducible.
Without `OPENAI_API_KEY`, the app returns a cited extractive response. To enable LLM
synthesis, set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in the environment.

## Technical choices

The preprocessing layer maps historical CFPB labels into four stable business
categories: Credit Card, Personal Loan, Savings Account, and Money Transfer. It
removes rows without narratives, normalizes case and whitespace, strips URLs and a
narrow set of complaint-introduction boilerplate, and preserves punctuation and
financial symbols that can carry meaning. The script records product distributions,
missing-narrative counts, length statistics, and short/long outliers in JSON and a
two-panel EDA plot. Actual findings are not stated here because the source dataset is
not committed; running the command above produces them without fabricated results.

For the learning-scale index, the default sample is 12,000 complaints allocated in
proportion to product frequency with a fixed random seed. Narratives are divided into
500-character chunks with 50-character overlap at word boundaries. This balances
specific retrieval against enough local context and matches the supplied full-scale
store specification. `all-MiniLM-L6-v2` is a compact 384-dimensional sentence model
with a practical quality/latency tradeoff for local semantic search. Embeddings are
L2-normalized and stored in `IndexFlatIP`, making inner product equivalent to cosine
similarity. Every chunk retains complaint, product, issue, company, date, and chunk
position metadata for auditability.

## Grounding and trust

The prompt tells the generator to use only retrieved evidence, cite numbered sources,
avoid unsupported counts or causal claims, and say when evidence is insufficient.
The UI exposes every source excerpt and similarity score beneath the answer. Product
filtering happens after an expanded candidate search so analysts can focus on one
service while still using the shared index. For deployment, use metadata-aware FAISS
shards or a filtered vector database when the corpus grows substantially.

## Qualitative evaluation

Run the following questions after building the full index and record results in
[`reports/evaluation.md`](reports/evaluation.md). A human reviewer should score
grounding, relevance, completeness, and citation correctness from 1 (poor) to 5
(excellent). Do not score fluency alone.

1. Why are customers unhappy with credit cards?
2. What problems prevent customers from accessing savings?
3. What are the main reasons money transfers are delayed?
4. Compare unexpected fees in personal loans and credit cards.
5. Are there complaints that suggest fraud or unauthorized transactions?
6. How do resolution difficulties differ across products?
7. What recurring issues should compliance investigate first?

Expected strengths are traceable evidence, cautious answers, and useful thematic
synthesis when the retrieved excerpts agree. Likely failure modes include retrieval
bias, duplicate chunks from one complaint, weak evidence for broad frequency claims,
and historical CFPB data not representing CrediTrust customers. Improvements include
date filters, complaint-level deduplication, reranking, hybrid keyword/vector search,
quantitative trend aggregation, and a labeled retrieval benchmark.

## Tests

```powershell
pytest -q
```

The CI workflow runs lightweight unit tests without downloading embedding models.

## Repository layout

```text
data/                 raw inputs and generated cleaned data (not committed)
notebooks/            notebook guidance
reports/              evaluation template and reporting assets
src/preprocess.py     Task 1 EDA and cleaning
src/index_complaints.py  Task 2 sampling, chunking, and FAISS indexing
src/rag.py            Task 3 retrieval, prompting, and generation
app.py                Task 4 Streamlit interface
tests/                deterministic unit tests
vector_store/         generated FAISS index and metadata (not committed)
```

