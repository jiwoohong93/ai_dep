"""Forward hooks on decoder-layer outputs: steering (caa / alphasteer) and activation capture.

Layer l refers to the *output* residual stream of decoder layer l (= HF hidden_states[l+1]).
Steering is applied at every token position. lambda == 0 short-circuits (no hook registered),
so logits are bit-identical to the unhooked model.
"""
import torch

from common import decoder_layers


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


class Steerer:
    """spec: dict layer -> ("alphasteer", M) with h <- h + lam * h @ M  (row convention, M = P @ tilde_delta)
                       or ("alphasteer_lr", (w, r)) same with M = w r^T (rank-1 factors, see alphasteer.py)
                       or ("caa", v)       with h <- h + lam * v      (v = r_hat * mean_norm_l).

    mode (AlphaSteer kinds only; CAA always adds its constant vector at every position):
          "per_token"      - every position t gets its own Delta h_t (breaks the model: the null space is fit on last-token activations only)
          "last_broadcast" - official AlphaSteer code: on the prefill pass, Delta h of the last (left-padded)
                             position is added to all positions; decode steps (seq_len 1) are not steered.
    """

    def __init__(self, model, spec=None, lam=0.0, mode="per_token"):
        assert mode in ("per_token", "last_broadcast")
        self.model, self.spec, self.lam, self.mode = model, spec or {}, float(lam), mode
        self.handles = []

    def __enter__(self):
        if self.lam == 0.0 or not self.spec:
            return self
        layers = decoder_layers(self.model)
        for l, (kind, mat) in self.spec.items():
            dev = next(layers[l].parameters()).device
            mat = tuple(m.to(dev, torch.float32) for m in mat) if kind == "alphasteer_lr" \
                else mat.to(dev, torch.float32)
            self.handles.append(layers[l].register_forward_hook(self._make(kind, mat)))
        return self

    def _make(self, kind, mat):
        lam, mode = self.lam, self.mode

        def hook(mod, inp, out):
            h = _hidden(out)
            if kind == "caa":  # input-independent vector: always every position (prefill + decode)
                return _replace(out, (h.float() + mat * lam).to(h.dtype))
            if mode == "last_broadcast":
                if h.shape[1] == 1:
                    return None
                src = h[:, -1:, :]
            else:
                src = h
            if kind == "alphasteer":
                delta = (src.float() @ mat) * lam
            elif kind == "alphasteer_lr":
                w, r = mat
                delta = (src.float() @ w).unsqueeze(-1) * (r * lam)
            else:
                raise ValueError(kind)
            return _replace(out, (h.float() + delta).to(h.dtype))  # broadcasts over positions
        return hook

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


class Capture:
    """Captures the last-position output of the given layers (after any steering hook)."""

    def __init__(self, model, layers):
        self.model, self.layers, self.store, self.handles = model, list(layers), {}, []

    def __enter__(self):
        mods = decoder_layers(self.model)
        for l in self.layers:
            self.handles.append(mods[l].register_forward_hook(self._make(l)))
        return self

    def _make(self, l):
        def hook(mod, inp, out):
            self.store[l] = _hidden(out)[:, -1, :].detach()
        return hook

    def stacked(self):
        """[L_sel, B, d]"""
        return torch.stack([self.store[l] for l in self.layers], 0)

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []
