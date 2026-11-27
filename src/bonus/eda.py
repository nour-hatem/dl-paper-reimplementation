"""Reusable exploratory data analysis helpers for ArSarcasm-v2.

The functions in this module return plain pandas objects or dictionaries so the
notebooks can print and plot the results without hiding any logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_TRAIN_PATH = Path("data/raw/arsarcasm_v2/training_data.csv")
RAW_TEST_PATH = Path("data/raw/arsarcasm_v2/testing_data.csv")
EXPECTED_COLUMNS = ("tweet", "sarcasm", "sentiment", "dialect")
SENTIMENT_ORDER = ("POS", "NEG", "NEU")
VALID_SENTIMENTS = frozenset(SENTIMENT_ORDER)
ARABERT_MODEL_NAME = "aubmindlab/bert-base-arabertv02"

# All figures use shades of blue only.
BLUE_DARK = "#1F4E79"
BLUE_MEDIUM = "#4C72B0"
BLUE_LIGHT = "#9DC3E6"

# Patterns used to measure how often URLs, mentions, and hashtags occur.
MARKER_PATTERNS = {
    "url": r"https?://\S+|www\.\S+",
    "mention": r"(?<!\w)@\w+",
    "hashtag": r"(?<!\w)#\w+",
}


def load_split(path: str | Path, required_columns: Iterable[str] = EXPECTED_COLUMNS) -> pd.DataFrame:
    """Load one raw CSV and verify that the expected columns are present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def check_sentiment_labels(frame: pd.DataFrame) -> dict:
    """Describe whether the sentiment column contains valid labels.

    Placeholder values are detected by comparing the normalised column against
    the three labels used by ArSarcasm-v2 (POS, NEG, NEU). The verdict is
    neutral and does not assume which split the frame represents.
    """
    if "sentiment" not in frame.columns:
        return {
            "has_column": False,
            "rows": len(frame),
            "non_null": 0,
            "empty": len(frame),
            "invalid_values": [],
            "observed_labels": [],
            "is_usable": False,
            "verdict": "MISSING SENTIMENT COLUMN",
        }

    raw = frame["sentiment"]
    normalised = raw.astype(str).str.strip().str.upper()
    non_null_mask = raw.notna()
    empty_mask = non_null_mask & normalised.eq("")
    invalid_mask = non_null_mask & ~empty_mask & ~normalised.isin(VALID_SENTIMENTS)
    observed = sorted(normalised[non_null_mask & ~empty_mask].unique().tolist())

    is_usable = non_null_mask.all() and not empty_mask.any() and not invalid_mask.any()
    verdict = "VALID SENTIMENT LABELS" if is_usable else "EMPTY OR PLACEHOLDER LABELS"

    return {
        "has_column": True,
        "rows": len(frame),
        "non_null": int(non_null_mask.sum()),
        "empty": int(empty_mask.sum()),
        "invalid_values": sorted(normalised[invalid_mask].unique().tolist()),
        "observed_labels": observed,
        "is_usable": bool(is_usable),
        "verdict": verdict,
    }


def sentiment_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sentiment counts and percentages in the paper's label order."""
    labels = frame["sentiment"].astype(str).str.strip().str.upper()
    counts = labels.value_counts()
    order = [label for label in SENTIMENT_ORDER if label in counts.index]
    order += [label for label in counts.index if label not in order]

    result = counts.reindex(order).to_frame(name="count")
    result["percentage"] = 100.0 * result["count"] / len(frame)
    return result


def text_lengths(frame: pd.DataFrame, text_column: str = "tweet") -> pd.DataFrame:
    """Return character and whitespace-token lengths for every tweet."""
    text = frame[text_column].astype(str)
    return pd.DataFrame(
        {
            "chars": text.str.len(),
            "whitespace_tokens": text.str.split().str.len(),
        }
    )


def tokenizer_lengths(
    texts: Iterable[str],
    tokenizer,
    batch_size: int = 64,
) -> np.ndarray:
    """Return the AraBERT token count for every text, including special tokens."""
    texts = list(texts)
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
    return np.asarray(lengths, dtype=int)


def summarize_token_lengths(lengths: Iterable[int], max_length: int = 256) -> dict:
    """Summarise precomputed token lengths against a maximum length limit."""
    lengths_array = np.asarray(list(lengths), dtype=int)
    over_limit = int((lengths_array > max_length).sum())
    return {
        "n_texts": len(lengths_array),
        "max_length_limit": max_length,
        "max_observed": int(lengths_array.max()) if len(lengths_array) else 0,
        "mean_observed": float(lengths_array.mean()) if len(lengths_array) else 0.0,
        "median_observed": float(np.median(lengths_array)) if len(lengths_array) else 0.0,
        "over_limit_count": over_limit,
        "over_limit_percentage": 100.0 * over_limit / len(lengths_array) if len(lengths_array) else 0.0,
    }


def tokenizer_length_summary(
    texts: Iterable[str],
    tokenizer,
    max_length: int = 256,
    batch_size: int = 64,
) -> dict:
    """Tokenize texts without truncation and summarise lengths against a limit."""
    lengths = tokenizer_lengths(texts, tokenizer, batch_size=batch_size)
    return summarize_token_lengths(lengths, max_length=max_length)


def marker_breakdown(
    frame: pd.DataFrame,
    token_lengths: Iterable[int],
    text_column: str = "tweet",
) -> pd.DataFrame:
    """Compare marker prevalence and tokens-per-word with and without each marker.

    Ratios are aggregate ratios: total AraBERT tokens divided by total
    whitespace tokens over the relevant subset of tweets.
    """
    text = frame[text_column].astype(str)
    word_counts = text.str.split().str.len().to_numpy()
    token_lengths = np.asarray(list(token_lengths), dtype=int)
    if len(token_lengths) != len(frame):
        raise ValueError("token_lengths must contain one value per row of frame")

    records = []
    for marker, pattern in MARKER_PATTERNS.items():
        mask = text.str.contains(pattern, regex=True).to_numpy()
        with_ratio = (
            token_lengths[mask].sum() / word_counts[mask].sum() if mask.any() else float("nan")
        )
        without_mask = ~mask
        without_ratio = (
            token_lengths[without_mask].sum() / word_counts[without_mask].sum()
            if without_mask.any()
            else float("nan")
        )
        records.append(
            {
                "marker": marker,
                "tweet_count": int(mask.sum()),
                "share_pct": 100.0 * mask.mean(),
                "tokens_per_word_with": float(with_ratio),
                "tokens_per_word_without": float(without_ratio),
            }
        )
    return pd.DataFrame(records)


def token_statistics(
    frame: pd.DataFrame,
    text_column: str,
    tokenizer,
    max_length: int = 256,
) -> dict:
    """Return token-limit, length, and tokens-per-word statistics for one split."""
    text = frame[text_column].astype(str)
    lengths = tokenizer_lengths(text.tolist(), tokenizer)
    word_counts = text.str.split().str.len().to_numpy()
    summary = summarize_token_lengths(lengths, max_length=max_length)
    summary["mean_tokens"] = float(lengths.mean()) if len(lengths) else 0.0
    summary["mean_words"] = float(word_counts.mean()) if len(word_counts) else 0.0
    per_tweet_ratio = lengths[word_counts > 0] / word_counts[word_counts > 0]
    summary["mean_tweet_tokens_per_word"] = (
        float(per_tweet_ratio.mean()) if len(per_tweet_ratio) else 0.0
    )
    summary["tokens_per_word"] = (
        float(lengths.sum() / word_counts.sum()) if word_counts.sum() else 0.0
    )
    return summary


def duplicate_summary(frame: pd.DataFrame, text_column: str = "tweet") -> dict:
    """Count exact duplicate rows and duplicate tweets (ignoring other columns)."""
    return {
        "full_row_duplicates": int(frame.duplicated().sum()),
        "tweet_duplicates": int(frame.duplicated(subset=[text_column]).sum()),
        "unique_tweets": int(frame[text_column].nunique()),
    }


def missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Report NaN values and empty strings for every column."""
    records = []
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        empty = int((series.dropna().astype(str).str.strip() == "").sum())
        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing": missing,
                "empty_strings": empty,
                "missing_or_empty_pct": 100.0 * (missing + empty) / len(frame),
            }
        )
    return pd.DataFrame(records)


def normalize_whitespace(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal whitespace runs."""
    return " ".join(str(text).split())


def tweet_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    text_column: str = "tweet",
    normalize: bool = False,
) -> dict:
    """Measure tweet-text overlap between the train and test files.

    Matching is exact unless normalize=True, in which case leading/trailing
    whitespace is stripped and internal whitespace runs are collapsed.
    """
    train_texts = train[text_column].astype(str)
    test_texts = test[text_column].astype(str)
    if normalize:
        train_texts = train_texts.map(normalize_whitespace)
        test_texts = test_texts.map(normalize_whitespace)

    train_set = set(train_texts)
    test_set = set(test_texts)
    shared = train_set & test_set
    return {
        "train_unique": len(train_set),
        "test_unique": len(test_set),
        "overlap_count": len(shared),
        "overlap_percentage_of_test": 100.0 * len(shared) / len(test_set) if test_set else 0.0,
        "examples": sorted(shared)[:5],
    }


def conflicting_sentiment_duplicates(
    frame: pd.DataFrame,
    text_column: str = "tweet",
    label_column: str = "sentiment",
    max_examples: int = 5,
) -> dict:
    """Find repeated tweets whose sentiment labels disagree.

    Duplicate detection uses exact tweet text, matching duplicate_summary.
    """
    labels = frame[label_column].astype(str).str.strip().str.upper()
    grouped = frame.assign(_sentiment=labels).groupby(text_column, dropna=False)["_sentiment"]
    counts = grouped.agg(["count", "nunique"]).rename(
        columns={"count": "row_count", "nunique": "label_count"}
    )
    conflicts = counts[(counts["row_count"] > 1) & (counts["label_count"] > 1)]

    examples = []
    for tweet in conflicts.index[:max_examples]:
        label_counts = labels[frame[text_column] == tweet].value_counts()
        examples.append(
            {
                "tweet": tweet,
                "labels": ", ".join(
                    f"{label}={count}" for label, count in label_counts.items()
                ),
            }
        )

    return {
        "conflicting_tweet_count": int(len(conflicts)),
        "conflicting_row_count": int(conflicts["row_count"].sum()),
        "examples": examples,
    }


def dialect_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Return dialect counts and percentages sorted from most to least frequent."""
    counts = frame["dialect"].astype(str).str.strip().value_counts()
    result = counts.to_frame(name="count")
    result["percentage"] = 100.0 * result["count"] / len(frame)
    return result


def plot_text_length_histograms(lengths: pd.DataFrame, bins: int = 40) -> plt.Figure:
    """Plot character and whitespace-token length histograms side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    specifications = (
        ("chars", "Characters per tweet", BLUE_MEDIUM),
        ("whitespace_tokens", "Whitespace tokens per tweet", BLUE_LIGHT),
    )
    for axis, (column, title, color) in zip(axes, specifications):
        axis.hist(lengths[column], bins=bins, color=color, edgecolor="white")
        axis.set_title(title)
        axis.set_xlabel(title.split(" per ")[0])
        axis.set_ylabel("Number of tweets")
        axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_category_counts(counts: pd.Series, title: str, color: str = BLUE_DARK) -> plt.Figure:
    """Plot a simple bar chart for a categorical count series."""
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(counts.index.astype(str), counts.to_numpy(), color=color)
    axis.set_title(title)
    axis.set_ylabel("Number of tweets")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
