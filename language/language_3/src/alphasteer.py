"""AlphaSteer null-space-constrained steering (Sheng et al. 2025), our own implementation.

Conventions (match the official code, row vectors): activations h are rows, the steering matrix
M = P @ tilde_delta is applied as h <- h + lam * h @ M.  In column notation,
Delta = M^T, i.e. Delta h = M^T h.

Official code (src/utils/steering_utils.py):
    P   = Q Q^T, Q = last `ratio*d` right-singular vectors of H_b^T H_b      (smallest eigvals)
    X   = H_m P ;  A = X^T X + alpha P^T P ;  b = X^T R  (R = r repeated per row)
    tilde_delta = pinv(A) b ;  M = P tilde_delta
"""
import torch


def benign_cov(chunks, d, device="cpu"):
    """C = sum_i h_i h_i^T accumulated in fp64 over an iterable of [n, d] chunks."""
    C = torch.zeros(d, d, dtype=torch.float64, device=device)
    n = 0
    for x in chunks:
        x = x.to(device, torch.float64)
        C += x.T @ x
        n += x.shape[0]
    return C, n


def null_basis(C, ratio):
    """Orthonormal basis U [d, k] of the k = int(ratio*d) smallest-eigenvalue directions of C."""
    evals, evecs = torch.linalg.eigh(C)  # ascending
    k = int(C.shape[0] * ratio)
    return evecs[:, :k].contiguous(), evals


def solve_tilde_delta(H_m, U, r, alpha):
    """Closed-form ridge solution, returned in rank-1 factored form tilde_delta = w r^T.

    With P = U U^T (U orthonormal), A = U (U^T H^T H U + alpha I) U^T, so
    pinv(A) = U (G + alpha I)^{-1} U^T with G = U^T H^T H U, and b = X^T R = P (sum_i h_i) r^T.
    Hence tilde_delta = U (G + alpha I)^{-1} U^T (sum_i h_i) r^T = w r^T and M = P w r^T = w r^T.
    """
    H = H_m.to(torch.float64)
    U = U.to(H.device, torch.float64)
    r = r.to(H.device, torch.float64)
    HU = H @ U
    G = HU.T @ HU
    Gr = G + alpha * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    s = HU.sum(0)                                   # U^T sum_i h_i
    w = U @ torch.linalg.solve(Gr, s)
    ev = torch.linalg.eigvalsh(Gr)
    info = {"cond": float(ev[-1] / ev[0]), "eig_min": float(ev[0]), "eig_max": float(ev[-1])}
    return w, r, info


def solve_tilde_delta_pinv(H_m, P, r, alpha):
    """Literal official formula (dense pinv), used in tests to validate solve_tilde_delta."""
    H = H_m.to(torch.float64)
    P = P.to(H.device, torch.float64)
    r = r.to(H.device, torch.float64)
    X = H @ P
    A = X.T @ X + alpha * (P.T @ P)
    b = X.T @ r.repeat(X.shape[0], 1)
    td = torch.linalg.pinv(A, hermitian=True) @ b
    return P @ td


def steering_matrix(w, r):
    return torch.outer(w, r)


def apply_delta(h, w, r):
    """Delta h for rows h: (h @ w) r."""
    return (h.to(w.dtype) @ w)[:, None] * r[None, :]
