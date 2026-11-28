"""Aggregate validation histories from the three-seed baseline reproduction."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


SEEDS = (42, 43, 44)
EXPECTED_EPOCHS = 10
PAPER_ACCURACY = 74.98
PAPER_WEIGHTED_F1 = 74.90
VALIDATION_SIZE = 1255
# Use the sample standard deviation (ddof=1) across seeds in all summary rows.
STD_KIND = "sample"


def _unique_epochs(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Keep the latest row for each epoch and return rows in epoch order."""
    if "epoch" not in frame.columns:
        raise ValueError(f"History has no epoch column: {source}")
    frame = frame.copy()
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="raise").astype(int)
    # The training loop rewrites an epoch when resuming; this also makes a
    # concatenated interrupted/resumed history safe to aggregate.
    frame = frame.drop_duplicates(subset=["epoch"], keep="last")
    return frame.sort_values("epoch").reset_index(drop=True)


def merge_history_rows(*frames: pd.DataFrame) -> pd.DataFrame:
    """Merge history fragments, keeping the newest row when epochs overlap."""
    if not frames:
        return pd.DataFrame(columns=["epoch"])
    return _unique_epochs(pd.concat(frames, ignore_index=True), source="merged history")


def _metric_row(history_row: pd.Series, seed: int, view: str) -> dict[str, Any]:
    """Convert one flattened training history row into a report row."""
    row: dict[str, Any] = {
        "seed": seed,
        "view": view,
        "epoch": int(history_row["epoch"]),
    }
    for metric in (
        "accuracy",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "macro_f_pn",
    ):
        column = f"validation_{metric}"
        if column not in history_row.index:
            raise ValueError(f"History is missing required column {column}")
        row[metric] = float(history_row[column])

    per_class: dict[str, dict[str, float]] = {}
    for column, value in history_row.items():
        prefix = "validation_"
        suffixes = ("_precision", "_recall", "_f1")
        if not isinstance(column, str) or not column.startswith(prefix):
            continue
        remainder = column[len(prefix) :]
        if remainder.startswith("weighted_"):
            continue
        suffix = next((item for item in suffixes if remainder.endswith(item)), None)
        if suffix is None:
            continue
        label = remainder[: -len(suffix)]
        per_class.setdefault(label, {})[suffix[1:]] = float(value)
    if not per_class:
        raise ValueError("History contains no flattened per-class validation metrics")
    for label, metrics in sorted(per_class.items()):
        for metric in ("precision", "recall", "f1"):
            if metric not in metrics:
                raise ValueError(f"Missing {metric} for validation class {label}")
            row[f"{label}_{metric}"] = metrics[metric]

    matrix_column = "validation_confusion_matrix"
    if matrix_column not in history_row.index:
        raise ValueError(f"History is missing required column {matrix_column}")
    matrix_value = history_row[matrix_column]
    row["confusion_matrix"] = (
        json.loads(matrix_value) if isinstance(matrix_value, str) else matrix_value
    )
    return row


def _history_for_seed(seed: int, output_dirs: dict[int, str | Path]) -> pd.DataFrame:
    path = Path(output_dirs[seed]) / "history.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing history for seed {seed}: {path}")
    return _unique_epochs(pd.read_csv(path), source=str(path))


def _summary_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add across-seed mean and sample standard deviation for each view."""
    summaries: list[dict[str, Any]] = []
    metrics = sorted(
        key
        for key, value in seed_rows[0].items()
        if key not in {"seed", "view", "epoch", "confusion_matrix"}
        and isinstance(value, (int, float))
    )
    for view in ("best", "last"):
        view_rows = [row for row in seed_rows if row["view"] == view]
        for statistic in ("mean", "std"):
            summary: dict[str, Any] = {
                "seed": statistic,
                "view": view,
                "epoch": None,
                "confusion_matrix": None,
            }
            for metric in metrics:
                values = [float(row[metric]) for row in view_rows]
                if statistic == "mean":
                    summary[metric] = sum(values) / len(values)
                else:
                    mean = sum(values) / len(values)
                    summary[metric] = math.sqrt(
                        sum((value - mean) ** 2 for value in values)
                        / (len(values) - 1)
                    )
            summaries.append(summary)
    return summaries


def _paper_comparison(best_mean: dict[str, Any]) -> dict[str, Any]:
    """Compare best-view means with paper values and an approximate accuracy SE."""
    accuracy = float(best_mean["accuracy"])
    weighted_f1 = float(best_mean["weighted_f1"])
    # This binomial standard error is a rough validation-set scale reference;
    # weighted F1 does not have a binomial sampling distribution.
    accuracy_se_points = 100.0 * math.sqrt(
        (PAPER_ACCURACY / 100.0)
        * (1.0 - PAPER_ACCURACY / 100.0)
        / VALIDATION_SIZE
    )
    accuracy_diff = abs(accuracy * 100.0 - PAPER_ACCURACY)
    weighted_f1_diff = abs(weighted_f1 * 100.0 - PAPER_WEIGHTED_F1)
    return {
        "paper_accuracy": PAPER_ACCURACY,
        "mean_accuracy_percent": accuracy * 100.0,
        "accuracy_absolute_difference_points": accuracy_diff,
        "paper_weighted_f1": PAPER_WEIGHTED_F1,
        "mean_weighted_f1_percent": weighted_f1 * 100.0,
        "weighted_f1_absolute_difference_points": weighted_f1_diff,
        "rough_one_se_points": accuracy_se_points,
        "accuracy_within_rough_one_se": accuracy_diff <= accuracy_se_points,
        "weighted_f1_within_rough_one_se_reference": weighted_f1_diff
        <= accuracy_se_points,
    }


def _markdown_table(
    seed_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], comparison: dict[str, Any]
) -> str:
    """Render seed metrics, mean/std rows, and paper comparison as Markdown."""
    metrics = [
        "accuracy",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "macro_f_pn",
    ]
    per_class = sorted(
        key
        for key in seed_rows[0]
        if not key.startswith("weighted_")
        if any(key.endswith(f"_{metric}") for metric in ("precision", "recall", "f1"))
    )
    columns = ["seed", "view", "epoch", *metrics, *per_class]
    rows = seed_rows + summary_rows
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None:
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            f"Sample standard deviation across seeds (ddof=1).",
            "",
            "## Confusion matrices",
            "",
        ]
    )
    for row in seed_rows:
        lines.append(
            f"- Seed {row['seed']} {row['view']} epoch {row['epoch']}: "
            f"`{json.dumps(row['confusion_matrix'], separators=(',', ':'))}`"
        )
    lines.extend(
        [
            "",
            "## Paper validation comparison",
            "",
            f"- Accuracy: mean {comparison['mean_accuracy_percent']:.2f}% vs "
            f"paper {PAPER_ACCURACY:.2f}%; absolute difference "
            f"{comparison['accuracy_absolute_difference_points']:.2f} points.",
            f"- Weighted F1: mean {comparison['mean_weighted_f1_percent']:.2f}% vs "
            f"paper {PAPER_WEIGHTED_F1:.2f}%; absolute difference "
            f"{comparison['weighted_f1_absolute_difference_points']:.2f} points.",
            f"- Rough one-standard-error reference for n={VALIDATION_SIZE}: "
            f"{comparison['rough_one_se_points']:.2f} percentage points (binomial "
            "accuracy approximation). Accuracy is "
            f"{'inside' if comparison['accuracy_within_rough_one_se'] else 'outside'} "
            "this reference; weighted F1 is "
            f"{'inside' if comparison['weighted_f1_within_rough_one_se_reference'] else 'outside'} "
            "the same rough scale reference (not a formal F1 standard error).",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate_runs(
    output_dirs: dict[int, str | Path],
    *,
    summary_csv: str | Path,
    summary_markdown: str | Path,
) -> dict[str, Any]:
    """Aggregate best and last validation epochs for every seed and save reports."""
    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        history = _history_for_seed(seed, output_dirs)
        expected = list(range(1, EXPECTED_EPOCHS + 1))
        actual = history["epoch"].tolist()
        if actual != expected:
            raise ValueError(
                f"History for seed {seed} must contain exactly epochs 1 to "
                f"{EXPECTED_EPOCHS}; found {actual}"
            )
        # Highest macro F-PN wins; ties select the earliest epoch. This matches
        # training checkpoint selection, which updates only on strict improvement.
        best_index = history.sort_values(
            ["validation_macro_f_pn", "epoch"],
            ascending=[False, True],
            kind="stable",
        ).index[0]
        best = _metric_row(history.loc[best_index], seed, "best")
        last = _metric_row(history.iloc[-1], seed, "last")
        seed_rows.extend((best, last))

    summary_rows = _summary_rows(seed_rows)
    best_mean = next(
        row for row in summary_rows if row["view"] == "best" and row["seed"] == "mean"
    )
    comparison = _paper_comparison(best_mean)
    csv_path = Path(summary_csv)
    markdown_path = Path(summary_markdown)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_rows = seed_rows + summary_rows
    # Matrices remain JSON text in CSV so the file has a single rectangular schema.
    csv_rows = [
        {
            key: json.dumps(value) if key == "confusion_matrix" and value is not None else value
            for key, value in row.items()
        }
        for row in report_rows
    ]
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    markdown_path.write_text(
        _markdown_table(seed_rows, summary_rows, comparison), encoding="utf-8"
    )
    return {
        "seed_rows": seed_rows,
        "summary_rows": summary_rows,
        "paper_comparison": comparison,
        "summary_csv": str(csv_path),
        "summary_markdown": str(markdown_path),
        "std_kind": STD_KIND,
    }


def run_self_test() -> None:
    """Check epoch selection, overlap-safe merging, and sample statistics."""
    with tempfile.TemporaryDirectory(prefix="baseline-aggregate-self-test-") as temp_dir:
        root = Path(temp_dir)
        columns = {
            "validation_accuracy": [0.5, 0.6],
            "validation_weighted_precision": [0.5, 0.6],
            "validation_weighted_recall": [0.5, 0.6],
            "validation_weighted_f1": [0.5, 0.6],
            "validation_macro_f_pn": [0.4, 0.8],
            "validation_NEG_precision": [0.5, 0.6],
            "validation_NEG_recall": [0.5, 0.6],
            "validation_NEG_f1": [0.5, 0.6],
            "validation_NEU_precision": [0.5, 0.6],
            "validation_NEU_recall": [0.5, 0.6],
            "validation_NEU_f1": [0.5, 0.6],
            "validation_POS_precision": [0.5, 0.6],
            "validation_POS_recall": [0.5, 0.6],
            "validation_POS_f1": [0.5, 0.6],
            "validation_confusion_matrix": ["[[1,0,0],[0,1,0],[0,0,1]]"] * 2,
        }
        fragment = pd.DataFrame({"epoch": [1, 2], **columns})
        resumed_fragment = fragment.iloc[[1]].copy()
        resumed_fragment.loc[:, "validation_accuracy"] = 0.75
        interrupted_path = root / "interrupted_history.csv"
        resumed_path = root / "resumed_history.csv"
        fragment.iloc[[0]].to_csv(interrupted_path, index=False)
        resumed_fragment.to_csv(resumed_path, index=False)
        merged = merge_history_rows(
            pd.read_csv(interrupted_path), pd.read_csv(resumed_path)
        )
        assert merged["epoch"].tolist() == [1, 2]
        assert not merged["epoch"].duplicated().any()
        assert merged.loc[merged["epoch"] == 2, "validation_accuracy"].item() == 0.75

        output_dirs: dict[int, Path] = {}
        for seed, accuracy, f1 in ((42, 0.5, 0.4), (43, 0.75, 0.8), (44, 1.0, 0.6)):
            run_dir = root / str(seed)
            run_dir.mkdir()
            output_dirs[seed] = run_dir
            rows = []
            for epoch in range(1, EXPECTED_EPOCHS + 1):
                score = 0.9 if epoch == EXPECTED_EPOCHS else 0.1
                metric = accuracy
                row = {key: values[0] for key, values in columns.items()}
                row.update(
                    {
                        "epoch": epoch,
                        "validation_accuracy": metric,
                        "validation_weighted_precision": metric,
                        "validation_weighted_recall": metric,
                        "validation_weighted_f1": f1,
                        "validation_macro_f_pn": score,
                        "validation_confusion_matrix": "[[1,0,0],[0,1,0],[0,0,1]]",
                    }
                )
                rows.append(row)
            pd.DataFrame(rows).to_csv(run_dir / "history.csv", index=False)
        result = aggregate_runs(
            output_dirs,
            summary_csv=root / "summary.csv",
            summary_markdown=root / "summary.md",
        )
        best_rows = [row for row in result["seed_rows"] if row["view"] == "best"]
        assert [row["epoch"] for row in best_rows] == [EXPECTED_EPOCHS] * 3
        last_rows = [row for row in result["seed_rows"] if row["view"] == "last"]
        assert [row["epoch"] for row in last_rows] == [EXPECTED_EPOCHS] * 3
        best_mean = next(
            row
            for row in result["summary_rows"]
            if row["view"] == "best" and row["seed"] == "mean"
        )
        best_std = next(
            row
            for row in result["summary_rows"]
            if row["view"] == "best" and row["seed"] == "std"
        )
        assert math.isclose(best_mean["accuracy"], 0.75, abs_tol=1e-12)
        assert math.isclose(best_std["accuracy"], 0.25, abs_tol=1e-12)

        incomplete_dir = output_dirs[42]
        complete_history = pd.read_csv(incomplete_dir / "history.csv")
        complete_history.iloc[:-1].to_csv(incomplete_dir / "history.csv", index=False)
        try:
            aggregate_runs(
                output_dirs,
                summary_csv=root / "incomplete.csv",
                summary_markdown=root / "incomplete.md",
            )
        except ValueError as error:
            assert "exactly epochs 1 to" in str(error)
        else:
            raise AssertionError("Incomplete histories must raise ValueError")
