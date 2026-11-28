"""Model and loss factories for reproducible sentiment experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification


@dataclass
class ModelBundle:
    """Keep model construction and loss selection outside the training loop."""

    model: nn.Module
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


ModelFactory = Callable[[str, int], nn.Module]
LossFactory = Callable[..., Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]

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


class FocalLoss(nn.Module):
    """Multiclass focal loss with mean reduction and no class weights."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = torch.log_softmax(logits, dim=-1)
        log_pt = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (-((1.0 - log_pt.exp()) ** self.gamma) * log_pt).mean()


def _focal_loss_factory(*, gamma: float = 2.0, **_: object) -> nn.Module:
    return FocalLoss(gamma=gamma)


def _weighted_ce_factory(*, class_counts: Sequence[int] | None = None, **_: object) -> nn.Module:
    if class_counts is None or len(class_counts) == 0 or any(int(value) <= 0 for value in class_counts):
        raise ValueError("class_weighted_ce requires positive training-split class counts")
    counts = torch.tensor(class_counts, dtype=torch.float64)
    weights = counts.sum() / (len(counts) * counts)
    return nn.CrossEntropyLoss(weight=weights.to(dtype=torch.float32))


class EncoderHead(nn.Module):
    """BERT encoder with masked mean pooling or a configurable MLP head."""

    def __init__(self, model_name: str, num_labels: int, *, richer: bool, hidden_size: int = 256,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.encoder = (
            AutoModel.from_config(model_name)
            if not isinstance(model_name, str)
            else AutoModel.from_pretrained(model_name)
        )
        self.dropout = nn.Dropout(dropout)
        self.richer = richer
        if richer:
            activations = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
            if activation not in activations:
                raise ValueError(f"Unsupported head activation: {activation}")
            self.classifier = nn.Sequential(
                nn.Linear(self.encoder.config.hidden_size, hidden_size),
                activations[activation](),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels),
            )
        else:
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.encoder.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        inputs.update(kwargs)
        encoded = self.encoder(**inputs)
        mask = attention_mask.unsqueeze(-1).to(encoded.last_hidden_state.dtype)
        pooled = (encoded.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.dropout(pooled))
        return SimpleNamespace(logits=logits)


def _encoder_head_factory(model_name: str, num_labels: int, **params: object) -> nn.Module:
    return EncoderHead(model_name, num_labels, richer=False, dropout=float(params.get("dropout", 0.1)))


def _richer_head_factory(model_name: str, num_labels: int, **params: object) -> nn.Module:
    return EncoderHead(model_name, num_labels, richer=True,
                       hidden_size=int(params.get("hidden_size", 256)),
                       dropout=float(params.get("dropout", 0.1)),
                       activation=str(params.get("activation", "relu")))


register_model_factory("mean_pooling", _encoder_head_factory)
register_model_factory("richer_head", _richer_head_factory)
register_loss_factory("class_weighted_ce", _weighted_ce_factory)
register_loss_factory("focal_loss", _focal_loss_factory)


def layerwise_parameter_groups(model: nn.Module, learning_rate: float, decay: float):
    """Return non-overlapping BERT parameter groups with layer-wise decay."""
    if not 0.0 < decay <= 1.0:
        raise ValueError("layer-wise decay must be in (0, 1]")
    encoder = getattr(model, "bert", None) or getattr(model, "roberta", None)
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    if layers is None:
        raise ValueError("Layer-wise learning rate decay requires a BERT encoder")
    layer_count = len(layers)
    grouped: dict[float, list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        exponent = 0
        if ".encoder.layer." in name:
            index = int(name.split(".encoder.layer.", 1)[1].split(".", 1)[0])
            exponent = layer_count - 1 - index
        elif ".embeddings." in name:
            exponent = layer_count
        grouped.setdefault(learning_rate * (decay ** exponent), []).append(parameter)
    return [{"params": params, "lr": rate} for rate, params in sorted(grouped.items())]


def build_model(
    model_name: str,
    num_labels: int,
    *,
    architecture: str = "sequence_classification",
    loss: str = "cross_entropy",
    model_params: dict[str, object] | None = None,
    loss_params: dict[str, object] | None = None,
    class_counts: Sequence[int] | None = None,
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
    loss_kwargs = dict(loss_params or {})
    if loss == "class_weighted_ce":
        loss_kwargs["class_counts"] = class_counts
    return ModelBundle(
        model=model_factory(model_name, num_labels, **(model_params or {})),
        loss_fn=loss_factory(**loss_kwargs),
    )
