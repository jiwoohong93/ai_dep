"""Fixed BIO Viterbi decoding; no gold labels or learned transition scores.

Method reference: https://aclanthology.org/2020.findings-emnlp.166/
The CLS/content/SEP/right-padding layout is the existing dataset.py contract.
"""

from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=32)
def _constraints(label_names):
    if (
        not label_names
        or len(set(label_names)) != len(label_names)
        or "O" not in label_names
    ):
        raise ValueError("BIO label names must be unique and include O")
    for label in label_names:
        if label == "O":
            continue
        if not isinstance(label, str) or label[:2] not in {"B-", "I-"} or not label[2:]:
            raise ValueError(f"Invalid BIO label: {label!r}")
        if label.startswith("I-") and "B-" + label[2:] not in label_names:
            raise ValueError(f"Missing begin label for {label}")
    transitions = np.zeros((len(label_names), len(label_names)), dtype=np.float32)
    starts = np.zeros(len(label_names), dtype=np.float32)
    for target, label in enumerate(label_names):
        if label.startswith("I-"):
            starts[target] = -np.inf
            for source, previous in enumerate(label_names):
                if previous not in ("B-" + label[2:], label):
                    transitions[source, target] = -np.inf
    transitions.flags.writeable = starts.flags.writeable = False
    return transitions, starts


def legal_bio_logits(logits, attention_mask, label_names):
    """Return scores whose argmax is the highest-scoring legal BIO path.

    Raw-logit sums and log-probability sums give the same maximizing path.
    Ties choose the first label ID, consistently with numpy.argmax.
    CLS, SEP and padding scores are preserved and never join the path.
    """
    label_names = tuple(label_names)
    transitions, starts = _constraints(label_names)
    if logits.ndim != 3 or attention_mask.shape != logits.shape[:2]:
        raise ValueError(
            "Expected logits [batch, length, labels] and matching attention_mask"
        )
    if logits.shape[2] != len(label_names) or not logits.is_floating_point():
        raise ValueError("Logit class count/dtype does not match BIO labels")
    if not torch.isfinite(logits).all():
        raise ValueError("Logits must be finite")
    mask = attention_mask.detach().cpu().numpy()
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("attention_mask must be binary")
    lengths = mask.sum(axis=1).astype(np.int64)
    if (lengths < 2).any() or not np.array_equal(
        mask, np.arange(mask.shape[1])[None, :] < lengths[:, None]
    ):
        raise ValueError(
            "Expected contiguous CLS/content/SEP followed by right padding"
        )
    emissions = logits.detach().cpu().float().numpy()
    result = logits.detach().cpu().clone()
    classes = np.arange(len(label_names))
    for row, length in enumerate(lengths):
        count = int(length) - 2
        if count == 0:
            continue
        scores = emissions[row, 1] + starts
        backpointers = []
        for emission in emissions[row, 2 : count + 1]:
            candidates = scores[:, None] + transitions
            back = candidates.argmax(axis=0)
            backpointers.append(back)
            scores = candidates[back, classes] + emission
        path = [int(scores.argmax())]
        for back in reversed(backpointers):
            path.append(int(back[path[-1]]))
        path.reverse()
        result[row, 1 : count + 1] = -torch.inf
        result[row, torch.arange(1, count + 1), torch.tensor(path)] = 0
    return result.to(logits.device)
