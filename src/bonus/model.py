"""Model and loss factories for reproducible sentiment experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


@dataclass
class ModelBundle:
    """Keep model construction and loss selection outside the training loop."""

    model: nn.Module
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


ModelFactory = Callable[[str, int], nn.Module]
LossFactory = Callable[[], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]

MODEL_FACTORIES: dict[str, ModelFactory] = {}
LOSS_FACTORIES: dict[str, LossFactory] = {}


def register_model_factory(name: str, factory: ModelFactory) -> None:
    """Register a model constructor without changing the trainer."""
    MODEL_FACTORIES[name] = factory


def register_loss_factory(name: str, factory: LossFactory) -> None:
    """Register a loss constructor without changing the trainer."""
    LOSS_FACTORIES[name] = factory


def _standard_sequence_classifier(model_name: str, num_labels: int) -> nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )


register_model_factory("sequence_classification", _standard_sequence_classifier)
register_loss_factory("cross_entropy", nn.CrossEntropyLoss)


def build_model(
    model_name: str,
    num_labels: int,
    *,
    architecture: str = "sequence_classification",
    loss: str = "cross_entropy",
) -> ModelBundle:
    """Build a model and loss through independently replaceable factories.

    Pooling, classifier heads, or losses can be changed by registering another
    factory and selecting its name in the configuration; the trainer stays the same.
    """
    try:
        model_factory = MODEL_FACTORIES[architecture]
    except KeyError as exc:
        raise ValueError(f"Unknown model architecture factory: {architecture}") from exc
    try:
        loss_factory = LOSS_FACTORIES[loss]
    except KeyError as exc:
        raise ValueError(f"Unknown loss factory: {loss}") from exc
    return ModelBundle(
        model=model_factory(model_name, num_labels),
        loss_fn=loss_factory(),
    )
