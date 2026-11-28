"""Validation-only screening and deferred Colab checks for Phase E."""

from __future__ import annotations

import math
import copy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from transformers import BertConfig, BertForSequenceClassification

from .model import FocalLoss, _weighted_ce_factory, build_model, layerwise_parameter_groups
from .utils import set_seed


BASE_CONFIG = Path("configs/bonus/repro_arabertv02_base.yaml")
BASE_DIR = Path("checkpoints/bonus/repro_arabertv02_base")
PHASE_D_SUMMARY = Path("checkpoints/bonus/repro_arabertv02_base_seeds_summary.csv")
SUMMARY_CSV = Path("checkpoints/bonus/enhancement_screening_summary.csv")
SUMMARY_MD = Path("checkpoints/bonus/enhancement_screening_summary.md")
CANDIDATES = {
    "mean_pooling": Path("configs/bonus/enh_mean_pooling.yaml"),
    "class_weighted_ce": Path("configs/bonus/enh_class_weighted_ce.yaml"),
    "focal_loss": Path("configs/bonus/enh_focal_loss.yaml"),
    "layerwise_lr_decay": Path("configs/bonus/enh_layerwise_lr_decay.yaml"),
    "richer_head": Path("configs/bonus/enh_richer_head.yaml"),
}
METRICS = ("macro_f_pn", "accuracy", "weighted_f1")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _best_history(directory: Path, label: str) -> dict[str, Any] | None:
    path = directory / "history.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    expected = list(range(1, 11))
    if "epoch" not in frame or len(frame) != 10:
        return None
    try:
        epochs = pd.to_numeric(frame["epoch"], errors="raise").astype(int).tolist()
    except (TypeError, ValueError):
        return None
    if sorted(epochs) != expected:
        return None
    required = {f"validation_{metric}" for metric in METRICS}
    if not required.issubset(frame.columns):
        raise ValueError(f"{label} history is missing metrics: {sorted(required - set(frame.columns))}")
    frame = frame.copy()
    frame["epoch"] = epochs
    row = frame.sort_values(["validation_macro_f_pn", "epoch"], ascending=[False, True]).iloc[0]
    return {"epoch": int(row["epoch"]), **{metric: float(row[f"validation_{metric}"]) for metric in METRICS}}


def _phase_d_std() -> dict[str, float]:
    if not PHASE_D_SUMMARY.is_file():
        raise FileNotFoundError(f"Phase D summary is required before screening: {PHASE_D_SUMMARY}")
    summary = pd.read_csv(PHASE_D_SUMMARY, dtype={"seed": str})
    selected = summary[(summary["seed"] == "std") & (summary["view"] == "best")]
    if selected.empty:
        raise ValueError("Phase D summary has no row with seed='std' and view='best'")
    row = selected.iloc[0]
    missing = [metric for metric in METRICS if metric not in row or pd.isna(row[metric])]
    if missing:
        raise ValueError(f"Phase D best-view standard-deviation row is missing: {missing}")
    return {metric: float(row[metric]) for metric in METRICS}


def screen_candidates() -> dict[str, Any]:
    """Create validation-only candidate ranking using Phase D noise at runtime."""
    phase_d_std = _phase_d_std()
    baseline = _best_history(BASE_DIR, "Baseline seed 42")
    if baseline is None:
        raise ValueError("Baseline seed 42 history must contain exactly epochs 1 to 10")
    records: list[dict[str, Any]] = []
    complete: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for name, config_path in CANDIDATES.items():
        config = _load_config(config_path)
        candidate = _best_history(Path(config["paths"]["output_dir"]), name)
        if candidate is None:
            incomplete.append(name)
            records.append({"candidate": name, "status": "incomplete"})
            continue
        record: dict[str, Any] = {"candidate": name, "status": "complete", "epoch": candidate["epoch"]}
        for metric in METRICS:
            gain = candidate[metric] - baseline[metric]
            scale = phase_d_std[metric]
            record[metric] = candidate[metric]
            record[f"gain_{metric}"] = gain
            record[f"gain_in_phase_d_std_{metric}"] = gain / scale if scale else math.inf
        record["macro_gain_inconclusive"] = record["gain_in_phase_d_std_macro_f_pn"] < 1.0
        records.append(record)
        complete.append(record)

    ranking = sorted(complete, key=lambda row: (-row["macro_f_pn"], row["candidate"]))
    decision = None
    if not incomplete:
        if len(ranking) >= 2 and ranking[0]["macro_f_pn"] - ranking[1]["macro_f_pn"] <= phase_d_std["macro_f_pn"]:
            decision = {
                "status": "approval_required",
                "proposal": f"Rerun {ranking[0]['candidate']} and {ranking[1]['candidate']} on seeds 43 and 44.",
            }
        elif ranking[0]["gain_macro_f_pn"] <= phase_d_std["macro_f_pn"]:
            decision = {"status": "no_clear_improvement"}
        else:
            decision = {"status": "selected", "candidate": ranking[0]["candidate"]}

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(SUMMARY_CSV, index=False)
    lines = ["# Phase E validation screening", "", "| Candidate | Status | Best epoch | Macro F-PN | Gain | Gain / Phase D SD | Accuracy | Weighted F1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        if row["status"] != "complete":
            lines.append(f"| {row['candidate']} | incomplete | | | | | | |")
            continue
        lines.append(
            f"| {row['candidate']} | complete | {row['epoch']} | {row['macro_f_pn']:.6f} | "
            f"{row['gain_macro_f_pn']:.6f}{' (worse than baseline)' if row['gain_macro_f_pn'] < 0 else ''} | "
            f"{row['gain_in_phase_d_std_macro_f_pn']:.3f}{' (worse than baseline)' if row['gain_macro_f_pn'] < 0 else ''} | "
            f"{row['accuracy']:.6f} | {row['weighted_f1']:.6f} |"
        )
    lines.extend(["", f"Phase D best-view standard deviations: `{phase_d_std}`.", ""])
    if incomplete:
        lines.append(f"Incomplete candidates: {', '.join(incomplete)}. No selection decision emitted.")
    elif decision["status"] == "approval_required":
        lines.append(f"Proposal: {decision['proposal']} This requires user approval.")
    elif decision["status"] == "no_clear_improvement":
        lines.append("No clear improvement: the top candidate did not beat the baseline by more than one Phase D standard deviation.")
    else:
        lines.append(f"Top validation candidate: {decision['candidate']}.")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"records": records, "ranking": ranking, "incomplete": incomplete, "decision": decision}


def check_baseline_equivalence() -> None:
    """Compare baseline factory construction with direct BERT and CE construction."""
    config = _load_config(BASE_CONFIG)["model"]
    set_seed(123)
    direct = BertForSequenceClassification.from_pretrained(config["name"], num_labels=3)
    direct_loss = torch.nn.CrossEntropyLoss()
    set_seed(123)
    bundle = build_model(config["name"], 3)
    assert list(direct.state_dict()) == list(bundle.model.state_dict())
    for name, value in direct.state_dict().items():
        assert value.shape == bundle.model.state_dict()[name].shape
        assert torch.equal(value, bundle.model.state_dict()[name])
    assert type(bundle.loss_fn) is type(direct_loss)
    assert bundle.loss_fn.__dict__ == direct_loss.__dict__


def check_candidate_configs() -> None:
    """Verify each full candidate config changes only its named experiment setting."""
    baseline = _load_config(BASE_CONFIG)
    intended = {
        "mean_pooling": ("model", {"architecture": "mean_pooling", "params": {"dropout": 0.1}}),
        "class_weighted_ce": ("model", {"loss": "class_weighted_ce"}),
        "focal_loss": ("model", {"loss": "focal_loss", "loss_params": {"gamma": 2.0}}),
        "layerwise_lr_decay": ("training", {"layerwise_lr_decay": 0.9}),
        "richer_head": ("model", {"architecture": "richer_head", "params": {"hidden_size": 256, "activation": "relu", "dropout": 0.1}}),
    }
    for name, path in CANDIDATES.items():
        candidate = _load_config(path)
        expected = copy.deepcopy(baseline)
        section, changes = intended[name]
        expected[section].update(copy.deepcopy(changes))
        expected["paths"]["output_dir"] = f"checkpoints/bonus/enh_{name}_seed42"
        assert candidate == expected, f"Unexpected config changes for {name}"


def check_loss_and_init_candidates() -> None:
    """Check loss formulas, initialization equivalence, and layer group rates."""
    config = _load_config(BASE_CONFIG)["model"]
    for loss_name, loss_params in (("class_weighted_ce", {}), ("focal_loss", {"gamma": 2.0}), ("cross_entropy", {})):
        set_seed(91)
        candidate = build_model(config["name"], 3, loss=loss_name, loss_params=loss_params,
                                class_counts=[5, 3, 2] if loss_name == "class_weighted_ce" else None)
        set_seed(91)
        baseline = build_model(config["name"], 3)
        for key, value in baseline.model.state_dict().items():
            assert torch.equal(value, candidate.model.state_dict()[key])
    weights = _weighted_ce_factory(class_counts=[5, 3, 2]).weight
    expected = torch.tensor([10 / 15, 10 / 9, 10 / 6])
    assert torch.allclose(weights, expected)
    logits = torch.tensor([[1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    targets = torch.tensor([0, 1])
    assert torch.allclose(FocalLoss(gamma=0)(logits, targets), torch.nn.functional.cross_entropy(logits, targets))
    set_seed(7)
    model = build_model(config["name"], 3).model
    groups = layerwise_parameter_groups(model, 1e-5, 0.9)
    parameters = [parameter for group in groups for parameter in group["params"]]
    assert len(parameters) == len({id(parameter) for parameter in parameters}) == len(list(model.parameters()))
    assert sorted(group["lr"] for group in groups) == sorted({1e-5 * 0.9 ** index for index in range(13)})


def check_model_candidates() -> None:
    """Check custom heads on a tiny random BERT and masked mean pooling behavior."""
    tiny = BertConfig(vocab_size=40, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=32, num_labels=3)
    # Factories accept the config object for local random checks without downloads.
    batch_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    for architecture in ("mean_pooling", "richer_head"):
        bundle = build_model(tiny, 3, architecture=architecture,
                             model_params={"hidden_size": 8, "dropout": 0.1})
        assert bundle.model(input_ids=batch_ids, attention_mask=mask).logits.shape == (2, 3)
    set_seed(13)
    model = build_model(tiny, 3, architecture="mean_pooling").model.eval()
    changed_padding = batch_ids.clone()
    changed_padding[:, 3] = 17
    with torch.inference_mode():
        first = model(input_ids=batch_ids, attention_mask=mask).logits
        second = model(input_ids=changed_padding, attention_mask=mask).logits
    assert torch.allclose(first, second, atol=1e-6)
