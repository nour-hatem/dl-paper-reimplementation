"""Metrics and explicit, one-time held-out test evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from .data import ID_TO_LABEL, LABEL_TO_ID


def calculate_metrics(labels: list[int] | np.ndarray, predictions: list[int] | np.ndarray) -> dict[str, Any]:
    """Calculate aggregate, per-class, and confusion-matrix metrics."""
    labels_array = np.asarray(labels, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    label_ids = list(sorted(ID_TO_LABEL))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        labels=label_ids,
        average="weighted",
        zero_division=0,
    )
    per_class = {
        ID_TO_LABEL[label_id]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label_id in enumerate(label_ids)
    }
    macro_f_pn = (per_class["POS"]["f1"] + per_class["NEG"]["f1"]) / 2.0
    matrix = confusion_matrix(labels_array, predictions_array, labels=label_ids)
    return {
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "macro_f_pn": float(macro_f_pn),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


class _TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int):
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = self.labels[index]
        return encoded


def evaluate_model_on_csv(
    model: torch.nn.Module,
    tokenizer: Any,
    csv_path: str | Path,
    *,
    batch_size: int,
    max_length: int,
    device: str | torch.device,
) -> dict[str, Any]:
    """Evaluate a model on a labeled CSV using dynamic per-batch padding."""
    frame = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("Evaluation CSV must contain 'text' and 'label' columns")
    labels = frame["label"].astype(int).tolist()
    if any(label not in LABEL_TO_ID.values() for label in labels):
        raise ValueError("Evaluation CSV contains a label outside the fixed mapping")
    dataset = _TextDataset(frame, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: _pad_batch(rows, tokenizer),
    )
    device = torch.device(device)
    model.to(device)
    model.eval()
    predictions: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**{key: value for key, value in batch.items() if key != "labels"})
            predictions.extend(output.logits.argmax(dim=-1).cpu().tolist())
    return calculate_metrics(labels, predictions)


def _pad_batch(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, torch.Tensor]:
    labels = torch.tensor([row.pop("labels") for row in rows], dtype=torch.long)
    padded = tokenizer.pad(rows, padding=True, return_tensors="pt")
    padded["labels"] = labels
    return padded


def evaluate_test_once(
    model: torch.nn.Module,
    tokenizer: Any,
    test_csv: str | Path,
    run_dir: str | Path,
    *,
    batch_size: int,
    max_length: int,
    device: str | torch.device,
) -> dict[str, Any]:
    """Explicitly evaluate the held-out test CSV once for a run directory.

    The marker is created before reading the CSV so repeated calls cannot silently
    turn test evaluation into a tuning loop, including after a partial failure.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    marker_path = run_dir / "test_evaluation.json"
    try:
        with marker_path.open("x", encoding="utf-8") as marker:
            json.dump({"status": "started", "test_csv": str(test_csv)}, marker)
    except FileExistsError as exc:
        raise RuntimeError(f"Test evaluation was already attempted for this run: {run_dir}") from exc

    metrics = evaluate_model_on_csv(
        model,
        tokenizer,
        test_csv,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    marker_path.write_text(
        json.dumps({"status": "complete", "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    return metrics
