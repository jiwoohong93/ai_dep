"""Unit tests. Real-data / GPU tests run when artifacts exist (GPU test needs RUN_GPU_TESTS=1)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pytest
import torch

from alphasteer import apply_delta, benign_cov, null_basis, solve_tilde_delta, solve_tilde_delta_pinv, steering_matrix
from common import load_config, run_dir

torch.manual_seed(0)


def _synthetic(d=64, nb=2000, nm=300, rank=24):
    basis = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]
    Hb = torch.randn(nb, rank, dtype=torch.float64) @ basis[:, :rank].T * 5 + 1e-3 * torch.randn(nb, d, dtype=torch.float64)
    Hm = torch.randn(nm, d, dtype=torch.float64)
    r = torch.randn(d, dtype=torch.float64)
    return Hb, Hm, r


def test_closed_form_matches_official_pinv():
    Hb, Hm, r = _synthetic()
    C, _ = benign_cov(Hb.split(256), Hb.shape[1])
    U, _ = null_basis(C, 0.6)
    w, rr, _ = solve_tilde_delta(Hm, U, r, 10.0)
    M = steering_matrix(w, rr)
    M_pinv = solve_tilde_delta_pinv(Hm, U @ U.T, r, 10.0)
    assert torch.allclose(M, M_pinv, rtol=1e-6, atol=1e-8 * M.abs().max())
    assert torch.allclose(apply_delta(Hm, w, rr), Hm @ M)


def test_null_space_kills_benign():
    Hb, _, _ = _synthetic()
    C, _ = benign_cov([Hb], Hb.shape[1])
    U, _ = null_basis(C, 0.6)
    leak = ((Hb @ U).norm(dim=1) / Hb.norm(dim=1))
    assert leak.max() < 1e-2


def test_delta_hm_aligned_with_r():
    _, Hm, r = _synthetic()
    Hb, _, _ = _synthetic()
    C, _ = benign_cov([Hb], Hb.shape[1])
    U, _ = null_basis(C, 0.6)
    w, rr, _ = solve_tilde_delta(Hm, U, r, 1.0)
    dm = apply_delta(Hm, w, rr).mean(0)
    assert torch.nn.functional.cosine_similarity(dm, r, 0) > 0.99


def test_template_split_no_leak_and_balance():
    p = ROOT / "data/processed/kobbq.parquet"
    if not p.exists():
        pytest.skip("run scripts/01_prepare_data.py first")
    kb = pd.read_parquet(p)
    assert (kb.groupby("template_id").split.nunique() == 1).all()
    for cat, g in kb.groupby("category"):
        assert set(g.split) == {"train", "val", "test"}, cat
    for r in ("unknown", "stereo", "anti"):
        share = kb[f"{r}_option_idx"].value_counts(normalize=True)
        assert (share - 1 / 3).abs().max() < 0.01


def test_real_data_diagnostics():
    """Reports ||P h_b||/||h_b|| (benign train/val) and cos(mean Delta h_m, r) from Phase-5 diagnostics."""
    cfg = load_config()
    f = run_dir(cfg, "results") / "alphasteer_diag.csv"
    if not f.exists():
        pytest.skip("run scripts/05_alphasteer.py first")
    d = pd.read_csv(f)
    print(d.groupby("p")[["leak_benign_train_mean", "leak_benign_val_mean"]].describe().T)
    # "~0" relative to a random direction, whose expected leak is sqrt(p)
    assert (d.leak_benign_train_mean < 0.2 * np.sqrt(d.p)).all()
    assert (d.leak_benign_val_mean < 0.25 * np.sqrt(d.p)).all()
    assert (d.cos_mean_Dh_m_r > 0.999).all()  # Delta h = (h.w) r is exactly parallel to r


@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1", reason="set RUN_GPU_TESTS=1")
def test_lambda_zero_bit_identical():
    from common import load_model
    from hooks import Steerer
    from scoring import forward_batch, letter_ids
    cfg = load_config()
    model, tok = load_model(cfg)
    texts = ["<|im_start|>user\n안녕하세요? A, B, C 중 하나로 답하세요.<|im_end|>\n<|im_start|>assistant\n",
             "<|im_start|>user\n1+1은? A: 2 B: 3 C: 4<|im_end|>\n<|im_start|>assistant\n"]
    ids = letter_ids(tok, "ABC")
    d = model.config.hidden_size
    spec = {10: ("alphasteer_lr", (torch.randn(d) * 1e-3, torch.randn(d))), 12: ("caa", torch.randn(d))}
    base = forward_batch(model, tok, texts, ids)["letter_logits"]
    with Steerer(model, spec, 0.0):
        zero = forward_batch(model, tok, texts, ids)["letter_logits"]
    with Steerer(model, spec, 0.0, "last_broadcast"):
        zero_lb = forward_batch(model, tok, texts, ids)["letter_logits"]
    with Steerer(model, spec, 0.5):
        steered = forward_batch(model, tok, texts, ids)["letter_logits"]
    with Steerer(model, spec, 0.5, "last_broadcast"):
        steered_lb = forward_batch(model, tok, texts, ids)["letter_logits"]
    after = forward_batch(model, tok, texts, ids)["letter_logits"]
    assert np.array_equal(base, zero) and np.array_equal(base, zero_lb)
    assert np.array_equal(base, after)  # hooks removed cleanly
    assert not np.array_equal(base, steered) and not np.array_equal(base, steered_lb)
