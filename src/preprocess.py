"""EDA and preprocessing for CFPB consumer complaint data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

NARRATIVE_COLUMN = "Consumer complaint narrative"
PRODUCT_COLUMN = "Product"
ID_COLUMN = "Complaint ID"

PRODUCT_MAP = {
    "credit card": "Credit Card",
    "credit card or prepaid card": "Credit Card",
    "consumer loan": "Personal Loan",
    "payday loan": "Personal Loan",
    "payday loan, title loan, or personal loan": "Personal Loan",
    "personal loan": "Personal Loan",
    "checking or savings account": "Savings Account",
    "bank account or service": "Savings Account",
    "savings account": "Savings Account",
    "money transfer": "Money Transfer",
    "money transfer, virtual currency, or money service": "Money Transfer",
}

BOILERPLATE = (
    r"\b(?:i am writing to (?:file|submit|make) (?:a )?complaint(?: about)?|"
    r"this (?:letter|message) is (?:to|regarding) (?:file )?(?:a )?complaint)\b"
)


def normalize_product(value: object) -> str | None:
    """Map historical CFPB product labels to the four business categories."""
    if pd.isna(value):
        return None
    return PRODUCT_MAP.get(str(value).strip().lower())


def clean_narrative(value: object) -> str:
    """Normalize a narrative while preserving words, numbers, and sentence marks."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(BOILERPLATE, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?$%'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_complaints(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter CFPB rows and return traceable, cleaned complaint records."""
    required = {NARRATIVE_COLUMN, PRODUCT_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = frame.copy()
    result["product_category"] = result[PRODUCT_COLUMN].map(normalize_product)
    result = result[result["product_category"].notna()].copy()
    result["cleaned_narrative"] = result[NARRATIVE_COLUMN].map(clean_narrative)
    result = result[result["cleaned_narrative"].str.len() > 0].copy()
    result["narrative_word_count"] = result["cleaned_narrative"].str.split().str.len()

    if ID_COLUMN not in result:
        result[ID_COLUMN] = result.index.astype(str)
    result = result.drop_duplicates(subset=[ID_COLUMN], keep="first")
    return result.reset_index(drop=True)


def build_eda_summary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    narratives = raw[NARRATIVE_COLUMN]
    has_narrative = narratives.notna() & narratives.astype(str).str.strip().ne("")
    lengths = cleaned["narrative_word_count"]
    return {
        "raw_rows": int(len(raw)),
        "with_narrative": int(has_narrative.sum()),
        "without_narrative": int((~has_narrative).sum()),
        "filtered_rows": int(len(cleaned)),
        "product_distribution": cleaned["product_category"].value_counts().to_dict(),
        "word_count": {
            "median": float(lengths.median()) if len(lengths) else 0,
            "p95": float(lengths.quantile(0.95)) if len(lengths) else 0,
            "minimum": int(lengths.min()) if len(lengths) else 0,
            "maximum": int(lengths.max()) if len(lengths) else 0,
            "very_short_under_10": int((lengths < 10).sum()),
            "very_long_over_1000": int((lengths > 1000).sum()),
        },
    }


def save_eda_plot(cleaned: pd.DataFrame, output: Path) -> None:
    """Save product and narrative-length charts for reporting."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = cleaned["product_category"].value_counts().index
    sns.countplot(data=cleaned, y="product_category", order=order, ax=axes[0])
    axes[0].set(title="Complaints by product", xlabel="Complaints", ylabel="")
    sns.histplot(cleaned["narrative_word_count"].clip(upper=1000), bins=50, ax=axes[1])
    axes[1].set(title="Narrative word count (clipped at 1,000)", xlabel="Words")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/filtered_complaints.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/eda_summary.json"))
    parser.add_argument("--plot", type=Path, default=Path("data/processed/eda.png"))
    args = parser.parse_args()

    raw = pd.read_csv(args.input, low_memory=False)
    cleaned = prepare_complaints(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    summary = build_eda_summary(raw, cleaned)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_eda_plot(cleaned, args.plot)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

