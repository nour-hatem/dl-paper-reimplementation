"""Data preparation for the ArSarcasm-v2 sentiment experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .eda import load_split
from .preprocessing import preprocess_text

TEXT_COLUMN = "tweet"
LABEL_COLUMN = "sentiment"
OUTPUT_COLUMNS = ("text", "label")

# Fixed mapping used by every experiment. IDs follow alphabetical label order.
LABEL_TO_ID = {"NEG": 0, "NEU": 1, "POS": 2}
ID_TO_LABEL = {label_id: label for label, label_id in LABEL_TO_ID.items()}

# Counts reported in Wadhawan (2021), Table 3.
PAPER_CLASS_COUNTS = {
    "train": {"NEG": 4139, "NEU": 5197, "POS": 1957},
    "validation": {"NEG": 482, "NEU": 550, "POS": 223},
}
PAPER_TOTALS = {"train": 11293, "validation": 1255}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and verify that it contains a mapping."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return config


def map_sentiment_labels(frame: pd.DataFrame, label_column: str = LABEL_COLUMN) -> pd.Series:
    """Return integer sentiment labels using the fixed LABEL_TO_ID mapping."""
    normalized = frame[label_column].astype(str).str.strip().str.upper()
    unknown = sorted(set(normalized) - set(LABEL_TO_ID))
    if unknown:
        raise ValueError(f"Unknown sentiment labels: {unknown}")
    return normalized.map(LABEL_TO_ID).astype(int)


def to_text_label_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Preprocess tweet text and return the two columns used by training."""
    labels = map_sentiment_labels(frame)
    return pd.DataFrame(
        {
            "text": frame[TEXT_COLUMN].map(preprocess_text),
            "label": labels,
        }
    )


def split_train_validation(
    raw_train: pd.DataFrame,
    seed: int,
    validation_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified train/validation split of the official training file."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    labels = map_sentiment_labels(raw_train)
    train_raw, validation_raw = train_test_split(
        raw_train,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return train_raw, validation_raw


def build_splits(
    raw_train_path: str | Path,
    raw_test_path: str | Path,
    seed: int,
    validation_fraction: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Build preprocessed train, validation, and test frames.

    The test file is preprocessed with the same function as the training file.
    Its labels are never used for splitting or model selection.
    """
    raw_train = load_split(raw_train_path)
    raw_test = load_split(raw_test_path)
    train_raw, validation_raw = split_train_validation(raw_train, seed, validation_fraction)

    return {
        "train": to_text_label_frame(train_raw),
        "validation": to_text_label_frame(validation_raw),
        "test": to_text_label_frame(raw_test),
        "raw_train": train_raw,
        "raw_validation": validation_raw,
        "raw_test": raw_test,
    }


def save_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save train, validation, and test CSVs with exactly the text,label columns."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name in ("train", "validation", "test"):
        frame = splits[name][list(OUTPUT_COLUMNS)]
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        paths[name] = path
    return paths


def label_mapping_table() -> pd.DataFrame:
    """Return the fixed label-to-integer mapping as a table."""
    return pd.DataFrame(
        [{"label": label, "label_id": label_id} for label, label_id in LABEL_TO_ID.items()]
    )


def class_count_table(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Return count and percentage for each sentiment label ID."""
    counts = frame["label"].value_counts()
    rows = []
    for label_id in sorted(ID_TO_LABEL):
        count = int(counts.get(label_id, 0))
        rows.append(
            {
                "split": split_name,
                "label": ID_TO_LABEL[label_id],
                "label_id": label_id,
                "count": count,
                "percentage": 100.0 * count / len(frame) if len(frame) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compare_to_paper(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Compare observed split sizes and class counts with Wadhawan (2021)."""
    rows = []
    for split_name, frame in (("train", train), ("validation", validation)):
        observed = frame["label"].value_counts().to_dict()
        for label, paper_count in PAPER_CLASS_COUNTS[split_name].items():
            label_id = LABEL_TO_ID[label]
            observed_count = int(observed.get(label_id, 0))
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "ours": observed_count,
                    "paper": paper_count,
                    "difference": observed_count - paper_count,
                }
            )
        rows.append(
            {
                "split": split_name,
                "label": "TOTAL",
                "ours": len(frame),
                "paper": PAPER_TOTALS[split_name],
                "difference": len(frame) - PAPER_TOTALS[split_name],
            }
        )
    return pd.DataFrame(rows)
