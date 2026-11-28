"""Plain PyTorch training loop for the AraBERT v02 baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .evaluate import (
    calculate_metrics,
    evaluate_model_on_csv_with_predictions,
)
from .model import build_model, layerwise_parameter_groups
from .utils import capture_rng_state, restore_rng_state, set_seed


class TextDataset(Dataset):
    """Tokenize one text at a time; padding is deferred to each batch."""

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


def _collate_batch(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, torch.Tensor]:
    labels = torch.tensor([row.pop("labels") for row in rows], dtype=torch.long)
    batch = tokenizer.pad(rows, padding=True, return_tensors="pt")
    batch["labels"] = labels
    return batch


def _read_frame(csv_path: str | Path, max_samples: int | None = None) -> pd.DataFrame:
    path_text = str(csv_path).replace("\\", "/").rstrip("/")
    if path_text.rsplit("/", 1)[-1].lower() == "test.csv":
        raise ValueError("Training is not allowed to read a file named test.csv")
    frame = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"CSV must contain 'text' and 'label' columns: {csv_path}")
    frame = frame[["text", "label"]].copy()
    frame["label"] = frame["label"].astype(int)
    if max_samples is not None:
        frame = frame.iloc[:max_samples].copy()
    if frame.empty:
        raise ValueError(f"CSV has no rows to use: {csv_path}")
    return frame


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if key not in {"per_class", "confusion_matrix"}
    }
    for label, values in metrics["per_class"].items():
        for metric, value in values.items():
            row[f"{prefix}_{label}_{metric}"] = value
    row[f"{prefix}_confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
    return row


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    best_macro_f_pn: float,
    loader_generator: torch.Generator,
    scaler: Any,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
        "best_macro_f_pn": best_macro_f_pn,
        "rng_state": capture_rng_state(),
        "loader_generator_state": loader_generator.get_state(),
        "scaler_state": scaler.state_dict(),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        # Keep RNG and DataLoader generator byte states on CPU; model and
        # optimizer state loaders move tensors to their parameter devices.
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_checkpoint_atomic(payload: dict[str, Any], path: Path) -> None:
    """Atomically replace a checkpoint using a temporary file beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_predictions_atomic(predictions: pd.DataFrame, path: Path) -> None:
    """Atomically write validation predictions beside the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        predictions.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_or_verify_config_snapshot(
    run_dir: Path,
    config: dict[str, Any],
    *,
    train_path: Path,
    validation_path: Path,
    total_epochs: int,
    max_train_samples: int | None,
    max_validation_samples: int | None,
    selected_device: torch.device,
    resume: bool,
) -> None:
    snapshot = {
        "git_commit_hash": _git_commit_hash(),
        "config": json.loads(json.dumps(config, default=str)),
        "resolved": {
            "train_csv": str(train_path.resolve()),
            "validation_csv": str(validation_path.resolve()),
            "output_dir": str(run_dir.resolve()),
            "epochs": total_epochs,
            "max_train_samples": max_train_samples,
            "max_validation_samples": max_validation_samples,
            "device": str(selected_device),
            "seed": int(config["training"]["seed"]),
        },
    }
    snapshot_path = run_dir / "config.json"
    if resume and snapshot_path.exists():
        existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if existing != snapshot:
            raise ValueError(
                "Run configuration does not match the saved config.json; "
                "refusing to resume with a mismatched config."
            )
    elif resume:
        print(
            f"No config snapshot exists at {snapshot_path}; the earlier run "
            "cannot be verified. Recording the current config for this resume."
        )
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_training(
    config: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path | None = None,
    train_csv: str | Path | None = None,
    validation_csv: str | Path | None = None,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    epochs: int | None = None,
    device: str | None = None,
    max_epochs_this_call: int | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    """Train or resume a model and return the complete per-epoch history.

    The interface accepts only the training and validation files. This keeps the
    held-out test data outside all training and model-selection paths.
    """
    if isinstance(config, (str, Path)):
        with Path(config).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    cfg = config
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    if train_cfg.get("scheduler") is not None:
        raise ValueError("scheduler must be null; scheduler support is not implemented")
    if train_cfg.get("warmup_steps", 0) != 0:
        raise ValueError("warmup_steps must be 0; warmup support is not implemented")
    paths_cfg = cfg.get("paths", {})
    train_path = Path(train_csv or data_cfg["train_file"])
    validation_path = Path(validation_csv or data_cfg["validation_file"])
    run_dir = Path(output_dir or paths_cfg["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    total_epochs = int(epochs if epochs is not None else train_cfg["epochs"])
    if total_epochs < 1:
        raise ValueError("epochs must be at least 1")

    seed = int(train_cfg["seed"])
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _write_or_verify_config_snapshot(
        run_dir,
        cfg,
        train_path=train_path,
        validation_path=validation_path,
        total_epochs=total_epochs,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        selected_device=selected_device,
        resume=resume,
    )
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    train_frame = _read_frame(train_path, max_train_samples)
    # _read_frame preserves the dataset's integer label IDs; data.py defines
    # NEG=0, NEU=1, POS=2, matching TextDataset's direct int conversion.
    validation_frame = _read_frame(validation_path, max_validation_samples)
    train_dataset = TextDataset(train_frame, tokenizer, int(train_cfg["max_length"]))
    validation_dataset = TextDataset(validation_frame, tokenizer, int(train_cfg["max_length"]))

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        generator=loader_generator,
        collate_fn=lambda rows: _collate_batch(rows, tokenizer),
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        collate_fn=lambda rows: _collate_batch(rows, tokenizer),
        num_workers=0,
    )

    bundle = build_model(
        model_cfg["name"],
        int(model_cfg["num_labels"]),
        architecture=model_cfg.get("architecture", "sequence_classification"),
        loss=model_cfg.get("loss", "cross_entropy"),
        model_params=model_cfg.get("params"),
        loss_params=model_cfg.get("loss_params"),
        class_counts=(
            torch.bincount(
                torch.tensor(train_frame["label"].astype(int).tolist(), dtype=torch.long),
                minlength=int(model_cfg["num_labels"]),
            ).tolist()
            if model_cfg.get("loss") == "class_weighted_ce"
            else None
        ),
    )
    model = bundle.model.to(selected_device)
    optimizer_cfg = train_cfg["optimizer"]
    optimizer_name = optimizer_cfg["name"].lower()
    if optimizer_name != "adam":
        raise ValueError("This baseline supports the configured Adam optimizer")
    learning_rate = float(optimizer_cfg["learning_rate"])
    decay = train_cfg.get("layerwise_lr_decay")
    optimizer_parameters = model.parameters()
    if decay is not None:
        decay = float(decay)
        optimizer_parameters = layerwise_parameter_groups(model, learning_rate, decay)
    optimizer = torch.optim.Adam(
        optimizer_parameters,
        lr=learning_rate,
        eps=float(optimizer_cfg["epsilon"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    mixed_precision = bool(train_cfg.get("mixed_precision", False))
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and selected_device.type == "cuda")

    start_epoch = 0
    history: list[dict[str, Any]] = []
    best_macro_f_pn = float("-inf")
    last_checkpoint = run_dir / "checkpoint_last.pt"
    if resume:
        if not last_checkpoint.exists():
            raise FileNotFoundError(f"No last checkpoint to resume: {last_checkpoint}")
        checkpoint = _load_checkpoint(last_checkpoint)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint["history"])
        best_macro_f_pn = float(checkpoint["best_macro_f_pn"])
        loader_generator.set_state(checkpoint["loader_generator_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        restore_rng_state(checkpoint["rng_state"])
        _write_history(run_dir / "history.csv", history)

    stop_epoch = total_epochs
    if max_epochs_this_call is not None:
        stop_epoch = min(stop_epoch, start_epoch + max_epochs_this_call)

    for epoch_index in range(start_epoch, stop_epoch):
        model.train()
        train_loss_sum = 0.0
        train_labels: list[int] = []
        train_predictions: list[int] = []
        for batch in train_loader:
            batch = {key: value.to(selected_device) for key, value in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(selected_device, mixed_precision):
                output = model(**batch)
                loss = bundle.loss_fn(output.logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach()) * labels.size(0)
            train_labels.extend(labels.detach().cpu().tolist())
            train_predictions.extend(output.logits.detach().argmax(dim=-1).cpu().tolist())

        train_metrics = calculate_metrics(train_labels, train_predictions)
        train_metrics["loss"] = train_loss_sum / len(train_dataset)
        validation_metrics = _run_validation(
            model,
            validation_loader,
            bundle.loss_fn,
            selected_device,
            mixed_precision,
        )
        epoch_number = epoch_index + 1
        row: dict[str, Any] = {"epoch": epoch_number}
        row.update(_flatten_metrics("train", train_metrics))
        row.update(_flatten_metrics("validation", validation_metrics))
        history = [item for item in history if int(item["epoch"]) != epoch_number]
        history.append(row)
        history.sort(key=lambda item: int(item["epoch"]))

        score = float(validation_metrics["macro_f_pn"])
        improved = score > best_macro_f_pn
        if improved:
            best_macro_f_pn = score
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch_number,
            history=history,
            best_macro_f_pn=best_macro_f_pn,
            loader_generator=loader_generator,
            scaler=scaler,
        )
        if improved:
            best_checkpoint = run_dir / "checkpoint_best.pt"
            _save_checkpoint_atomic(payload, best_checkpoint)
        if improved:
            _, validation_predictions = evaluate_model_on_csv_with_predictions(
                model,
                tokenizer,
                validation_path,
                batch_size=int(train_cfg["batch_size"]),
                max_length=int(train_cfg["max_length"]),
                device=selected_device,
                max_samples=max_validation_samples,
            )
            _write_predictions_atomic(
                validation_predictions,
                run_dir / "validation_predictions.csv",
            )
        _write_history(run_dir / "history.csv", history)
        # The last checkpoint is the epoch commit marker: publish it only after
        # all artifacts for this completed epoch have been written.
        _save_checkpoint_atomic(payload, last_checkpoint)
        print(
            f"epoch {epoch_number}/{total_epochs}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"validation_loss={validation_metrics['loss']:.4f} "
            f"validation_macro_f_pn={score:.4f}"
        )
    return pd.DataFrame(history)


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Any,
    device: torch.device,
    mixed_precision: bool,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            with _autocast_context(device, mixed_precision):
                output = model(**batch)
                loss = loss_fn(output.logits, labels)
            loss_sum += float(loss) * labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(output.logits.argmax(dim=-1).cpu().tolist())
    metrics = calculate_metrics(labels_all, predictions_all)
    metrics["loss"] = loss_sum / len(labels_all)
    return metrics


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    columns = list(history[0]) if history else ["epoch"]
    pd.DataFrame(history, columns=columns).to_csv(path, index=False)


def generate_validation_predictions_from_checkpoint(
    config: dict[str, Any] | str | Path = "configs/bonus/repro_arabertv02_base.yaml",
    *,
    output_dir: str | Path | None = None,
    output_csv: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Write validation predictions from an existing best checkpoint only."""
    if isinstance(config, (str, Path)):
        with Path(config).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    cfg = config
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    data_cfg = cfg["data"]
    paths_cfg = cfg.get("paths", {})
    run_dir = Path(output_dir or paths_cfg["output_dir"])
    checkpoint_path = run_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    bundle = build_model(
        model_cfg["name"],
        int(model_cfg["num_labels"]),
        architecture=model_cfg.get("architecture", "sequence_classification"),
        loss=model_cfg.get("loss", "cross_entropy"),
        model_params=model_cfg.get("params"),
        loss_params=model_cfg.get("loss_params"),
        # The inference path does not evaluate loss; positive placeholders let
        # the selected weighted-loss factory be constructed without train data.
        class_counts=[1] * int(model_cfg["num_labels"]),
    )
    model = bundle.model.to(selected_device)
    checkpoint = _load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    metrics, predictions = evaluate_model_on_csv_with_predictions(
        model,
        tokenizer,
        data_cfg["validation_file"],
        batch_size=int(training_cfg["batch_size"]),
        max_length=int(training_cfg["max_length"]),
        device=selected_device,
    )
    predictions_path = Path(output_csv or run_dir / "validation_predictions.csv")
    _write_predictions_atomic(predictions, predictions_path)
    return {"metrics": metrics, "predictions_csv": str(predictions_path)}


def run_smoke_checks(config_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Run the requested fast CPU checks without consulting held-out data."""
    from .evaluate import calculate_metrics

    started = time.perf_counter()
    expected = {
        "accuracy": 0.6,
        "weighted_precision": 0.7,
        "weighted_recall": 0.6,
        "weighted_f1": 0.6,
        "macro_f_pn": 7.0 / 12.0,
    }
    measured = calculate_metrics([0, 0, 1, 2, 2], [0, 1, 1, 2, 0])
    for key, value in expected.items():
        if not math.isclose(measured[key], value, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"Hand-calculated metric mismatch for {key}: {measured[key]}")
    if measured["confusion_matrix"] != [[1, 1, 0], [0, 1, 0], [1, 0, 1]]:
        raise AssertionError("Hand-calculated confusion matrix mismatch")

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # Start each smoke run with one epoch, then resume one run to a second epoch
    # to exercise restoration from a real interruption.
    config["training"]["epochs"] = 2
    config["training"]["mixed_precision"] = False
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    read_paths: list[str] = []
    original_read_csv = pd.read_csv

    def guarded_read_csv(path: Any, *args: Any, **kwargs: Any):
        resolved = str(Path(path).resolve())
        read_paths.append(resolved)
        if Path(resolved).name.lower() == "test.csv":
            raise AssertionError("Training attempted to read the held-out CSV")
        return original_read_csv(path, *args, **kwargs)

    pd.read_csv = guarded_read_csv
    try:
        try:
            _read_frame(root / "test.csv")
        except ValueError as exc:
            if "test.csv" not in str(exc):
                raise
        else:
            raise AssertionError("_read_frame did not reject a path named test.csv")

        history_a = run_training(
            config,
            output_dir=root / "same_seed_a",
            max_train_samples=200,
            max_validation_samples=200,
            epochs=2,
            device="cpu",
            max_epochs_this_call=1,
        )
        history_b = run_training(
            config,
            output_dir=root / "same_seed_b",
            max_train_samples=200,
            max_validation_samples=200,
            epochs=2,
            device="cpu",
            max_epochs_this_call=1,
        )
        if not history_a.equals(history_b):
            raise AssertionError("Two CPU training runs with the same seed differed")
        uninterrupted = run_training(
            config,
            output_dir=root / "uninterrupted_two_epochs",
            max_train_samples=200,
            max_validation_samples=200,
            epochs=2,
            device="cpu",
        )
        resumed = run_training(
            config,
            output_dir=root / "same_seed_a",
            max_train_samples=200,
            max_validation_samples=200,
            epochs=2,
            device="cpu",
            resume=True,
        )
        if resumed["epoch"].tolist() != [1, 2] or resumed["epoch"].duplicated().any():
            raise AssertionError("Resume created duplicate or unexpected history epochs")
        resumed_epoch_two = resumed.loc[resumed["epoch"] == 2].reset_index(drop=True)
        uninterrupted_epoch_two = uninterrupted.loc[
            uninterrupted["epoch"] == 2
        ].reset_index(drop=True)
        if not resumed_epoch_two.equals(uninterrupted_epoch_two):
            raise AssertionError(
                "Resumed epoch 2 metrics differed from uninterrupted CPU training"
            )
    finally:
        pd.read_csv = original_read_csv

    allowed = {
        str(Path(config["data"]["train_file"]).resolve()),
        str(Path(config["data"]["validation_file"]).resolve()),
    }
    if set(read_paths) - allowed:
        raise AssertionError(f"Training read an unexpected CSV: {set(read_paths) - allowed}")
    test_csv_reads = sum(
        Path(read_path).name.lower() == "test.csv" for read_path in read_paths
    )
    if test_csv_reads != 0:
        raise AssertionError(f"Training read test.csv {test_csv_reads} time(s)")
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(root / "same_seed_a" / "checkpoint_best.pt")],
        check=False,
    ).returncode == 0
    if not ignored:
        raise AssertionError("Checkpoint directory is not ignored by Git")

    return {
        "status": "passed",
        "runtime_seconds": time.perf_counter() - started,
        "training_samples": 200,
        "initial_epochs_per_run": 1,
        "resumed_epoch_count": 2,
        "runs": 2,
        "resume_history_epochs": resumed["epoch"].tolist(),
        "same_seed_histories_identical": True,
        "resume_epoch_two_matches_uninterrupted": True,
        "uninterrupted_history_epochs": uninterrupted["epoch"].tolist(),
        "test_csv_reads": test_csv_reads,
        "git_ignored_checkpoint": True,
        "hand_metrics": measured,
        "sample_run_metrics": history_a.iloc[0].to_dict(),
    }


def run_notebook_workflow(
    config_path: str | Path = "configs/bonus/repro_arabertv02_base.yaml",
    *,
    smoke_mode: bool = True,
) -> Any:
    """Notebook entry point; smoke mode is the safe default."""
    if smoke_mode:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        return run_smoke_checks(config_path, config["paths"]["smoke_output_dir"])
    return run_training(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bonus/repro_arabertv02_base.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    history = run_training(args.config, resume=args.resume)
    print(history.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
