# RAG Qualitative Evaluation

Complete this table against the built full-data vector store. Quote only short source
excerpts and retain complaint IDs so results remain auditable.

| Question | Generated answer | Retrieved sources (1–2) | Quality (1–5) | Comments / analysis |
|---|---|---|---:|---|
| Why are customers unhappy with credit cards? | _Run pipeline_ | _Complaint IDs_ | — | Check theme synthesis and citations. |
| What problems prevent customers from accessing savings? | _Run pipeline_ | _Complaint IDs_ | — | Check relevance and product filtering. |
| What are the main reasons money transfers are delayed? | _Run pipeline_ | _Complaint IDs_ | — | Distinguish evidence from frequency claims. |
| Compare unexpected fees in personal loans and credit cards. | _Run pipeline_ | _Complaint IDs_ | — | Both products should be represented. |
| Are there complaints that suggest fraud or unauthorized transactions? | _Run pipeline_ | _Complaint IDs_ | — | Avoid asserting confirmed fraud. |
| How do resolution difficulties differ across products? | _Run pipeline_ | _Complaint IDs_ | — | Check cross-product completeness. |
| What recurring issues should compliance investigate first? | _Run pipeline_ | _Complaint IDs_ | — | Recommendations must follow evidence. |

## Scoring rubric

- **5:** Fully grounded, directly relevant, complete, and correctly cited.
- **4:** Useful and grounded with a minor omission or imprecision.
- **3:** Partly useful; misses an important theme or includes weak support.
- **2:** Mostly irrelevant, incomplete, or poorly grounded.
- **1:** Unsupported, misleading, or fails to answer.

