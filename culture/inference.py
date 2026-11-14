"""Checkpoint loading and label-independent inference for the culture evaluator."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from legacy_decode import legal_bio_logits

KOBERT_REVISION = "38279184ba645e8c94d709fbe92eb5bcb47312c1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_equal(actual, expected, description):
    if actual != expected:
        raise ValueError(f"{description}: expected {expected!r}, got {actual!r}")


def write_result(path, result):
    """Exclusive creation prevents accidentally replacing previous evidence."""
    path = Path(path)
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def load_checkpoint(model_class, config, path):
    model = (
        model_class.from_config(config)
        if hasattr(model_class, "from_config")
        else model_class(config)
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not all(isinstance(k, str) for k in state):
        raise ValueError("Checkpoint must be a plain model state_dict")
    state = state.copy()
    removed = []
    for key in list(state):
        if key.endswith(".embeddings.position_ids") and key not in model.state_dict():
            expected = torch.arange(config.max_position_embeddings).unsqueeze(0)
            if not torch.equal(state[key], expected):
                raise ValueError(f"Unexpected legacy position_ids buffer: {key}")
            del state[key]
            removed.append(key)
    model.load_state_dict(state, strict=True)
    return model, removed


class InferenceTagger(torch.nn.Module):
    """Accept but ignore gold ``labels`` for the historical evaluate signature.

    Gold is never forwarded to the classifier or decoder. The zero loss is
    an unused compatibility value, not a measured validation loss.
    """

    def __init__(self, model, label_names, decoder, cls_token_id, sep_token_id):
        super().__init__()
        if decoder not in {"argmax", "legal_bio"}:
            raise ValueError(f"Unknown decoder: {decoder}")
        self.model = model
        self.label_names = tuple(label_names)
        self.decoder = decoder
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def forward(self, input_ids, attention_mask, labels=None):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids and attention_mask must have equal [batch, length] shapes"
            )
        lengths = attention_mask.sum(1).long()
        if (lengths < 2).any() or (lengths > input_ids.shape[1]).any():
            raise ValueError("Every input needs CLS and SEP tokens")
        expected_mask = (
            torch.arange(input_ids.shape[1], device=attention_mask.device)[None, :]
            < lengths[:, None]
        )
        if not torch.equal(attention_mask, expected_mask.to(attention_mask.dtype)):
            raise ValueError("Expected binary, contiguous attention with right padding")
        rows = torch.arange(len(input_ids), device=input_ids.device)
        if not (input_ids[:, 0] == self.cls_token_id).all():
            raise ValueError("Expected CLS at the first position")
        if not (input_ids[rows, lengths - 1] == self.sep_token_id).all():
            raise ValueError("Expected SEP at the last unpadded position")
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        if not torch.isfinite(logits).all():
            raise ValueError("Model returned non-finite logits")
        if self.decoder == "legal_bio":
            logits = legal_bio_logits(logits, attention_mask, self.label_names)
        return logits.new_zeros(()), logits
