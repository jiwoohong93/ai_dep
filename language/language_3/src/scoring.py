"""Batched forward passes with letter-logit scoring at the last prompt position."""
import numpy as np
import torch

from hooks import Capture


def letter_ids(tok, letters):
    ids = []
    for L in letters:
        t = tok.encode(L, add_special_tokens=False)
        assert len(t) == 1, (L, t)
        ids.append(t[0])
    return ids


def space_letter_ids(tok, letters):
    return [tok.encode(" " + L, add_special_tokens=False)[0] for L in letters]


def make_batches(lengths, max_tokens, max_bs=64):
    """Length-sorted batches under a padded-token budget. Returns list of index arrays."""
    order = np.argsort(-np.asarray(lengths), kind="stable")
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and (m * (len(cur) + 1) > max_tokens or len(cur) >= max_bs):
            batches.append(np.array(cur)); cur, m = [], lengths[i]
        cur.append(i); cur_max = m
    if cur:
        batches.append(np.array(cur))
    return batches


@torch.no_grad()
def forward_batch(model, tok, texts, letters_ids, space_ids=None, capture_layers=None):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    cap = Capture(model, capture_layers) if capture_layers else None
    if cap:
        cap.__enter__()
    try:
        logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, position_ids=pos,
                       use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
    finally:
        if cap:
            cap.__exit__()
    lp = torch.log_softmax(logits, -1)
    out = {"letter_logits": logits[:, letters_ids].cpu().numpy(),
           "p_letters": lp[:, letters_ids].exp().sum(-1).cpu().numpy(),
           "top_token": logits.argmax(-1).cpu().numpy()}
    if space_ids is not None:
        out["p_space_letters"] = lp[:, space_ids].exp().sum(-1).cpu().numpy()
    if cap:
        out["acts"] = cap.stacked().to(torch.bfloat16).cpu()  # [L_sel, B, d]
    return out


def score_texts(model, tok, texts, letters, max_tokens=24000, capture_layers=None, progress=None):
    """Score all texts; returns dict of arrays in the original order (+ acts [L_sel, N, d] if captured)."""
    ids, sids = letter_ids(tok, letters), space_letter_ids(tok, letters)
    lengths = [len(x) for x in tok(texts, add_special_tokens=False).input_ids]
    n = len(texts)
    res = {"letter_logits": np.zeros((n, len(letters)), np.float32), "p_letters": np.zeros(n, np.float32),
           "p_space_letters": np.zeros(n, np.float32), "top_token": np.zeros(n, np.int64),
           "n_tokens": np.asarray(lengths)}
    acts = None
    batches = make_batches(lengths, max_tokens)
    for bi, b in enumerate(batches):
        o = forward_batch(model, tok, [texts[i] for i in b], ids, sids, capture_layers)
        for k in ("letter_logits", "p_letters", "p_space_letters", "top_token"):
            res[k][b] = o[k]
        if capture_layers:
            if acts is None:
                acts = torch.zeros(len(capture_layers), n, o["acts"].shape[-1], dtype=torch.bfloat16)
            acts[:, torch.as_tensor(b)] = o["acts"]
        if progress:
            progress(bi + 1, len(batches))
    ll = res["letter_logits"]
    res["pred_idx"] = ll.argmax(-1)
    s = np.sort(ll, -1)
    res["margin"] = s[:, -1] - s[:, -2]
    if capture_layers:
        res["acts"] = acts
    return res


@torch.no_grad()
def score_with_prefix(model, tok, prefix, suffixes, letters, max_tokens=24000, progress=None, capture_layers=None):
    """Score prompts that share `prefix` (e.g. KMMLU k-shot turns) by reusing its KV cache.

    The prefix must end at a special-token boundary so that tokenize(prefix)+tokenize(suffix)
    equals tokenize(prefix+suffix). Steering hooks active in the caller apply to prefix and suffix.
    """
    from transformers import DynamicCache
    ids, sids = letter_ids(tok, letters), space_letter_ids(tok, letters)
    dev = model.device
    p_ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    P = p_ids.shape[1]
    pkv = model(input_ids=p_ids, use_cache=True).past_key_values
    legacy = [(pkv.layers[i].keys, pkv.layers[i].values) if hasattr(pkv, "layers") else pkv[i]
              for i in range(len(pkv))]
    lengths = [len(x) for x in tok(suffixes, add_special_tokens=False).input_ids]
    n = len(suffixes)
    res = {"letter_logits": np.zeros((n, len(letters)), np.float32), "p_letters": np.zeros(n, np.float32),
           "p_space_letters": np.zeros(n, np.float32), "top_token": np.zeros(n, np.int64),
           "n_tokens": np.asarray(lengths) + P}
    batches = make_batches([l + P for l in lengths], max_tokens)
    for bi, b in enumerate(batches):
        enc = tok([suffixes[i] for i in b], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        B = len(b)
        cache = DynamicCache()
        for li, (k, v) in enumerate(legacy):
            cache.update(k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous(), li)
        mask = torch.cat([torch.ones(B, P, dtype=enc.attention_mask.dtype, device=dev), enc.attention_mask], 1)
        pos = P + (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
        cap = Capture(model, capture_layers) if capture_layers else None
        if cap:
            cap.__enter__()
        try:
            logits = model(input_ids=enc.input_ids, attention_mask=mask, position_ids=pos, past_key_values=cache,
                           use_cache=True, logits_to_keep=1).logits[:, -1, :].float()
        finally:
            if cap:
                cap.__exit__()
        if cap:
            a = cap.stacked().to(torch.bfloat16).cpu()
            if "acts" not in res:
                res["acts"] = torch.zeros(len(capture_layers), n, a.shape[-1], dtype=torch.bfloat16)
            res["acts"][:, torch.as_tensor(b)] = a
        lp = torch.log_softmax(logits, -1)
        res["letter_logits"][b] = logits[:, ids].cpu().numpy()
        res["p_letters"][b] = lp[:, ids].exp().sum(-1).cpu().numpy()
        res["p_space_letters"][b] = lp[:, sids].exp().sum(-1).cpu().numpy()
        res["top_token"][b] = logits.argmax(-1).cpu().numpy()
        del cache
        if progress:
            progress(bi + 1, len(batches))
    ll = res["letter_logits"]
    res["pred_idx"] = ll.argmax(-1)
    s = np.sort(ll, -1)
    res["margin"] = s[:, -1] - s[:, -2]
    return res
