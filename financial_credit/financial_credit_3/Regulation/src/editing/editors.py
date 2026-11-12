"""ROME / MEMIT / EMMET / AlphaEdit — fc2(W₂) 가중치 update rule 4종.

공통 프레임: W' = W + Δ 가 편집 요청 키 K에 대해 목표 v*를 내도록 하되(efficacy),
일반 키 분포(C)·보존 키(K₀)를 최대한 건드리지 않는다(locality).

표기: W [d_out×d_in] (1024×4096), K [d_in×R], V* [d_out×R], b [d_out]
Res = V* − (W K + b·1ᵀ)  (요청별 잔차)

- ROME      : 요청을 하나씩 rank-1로 순차 삽입 (C⁻¹ 방향, 등식 충족)
- MEMIT     : 배치 최소제곱 (ridge: λC 보존항), 다층 분산은 잔차 균등 분할
- EMMET     : 배치 등식 제약 (모든 요청을 정확히 충족하는 최소-C-노름 해)
- AlphaEdit : MEMIT식 배치 해를 보존 키 null-space에 사영 (Δ K₀ ≈ 0 보장)
"""
import torch


def _sym_inv(A, eps=1e-4):
    """대칭 행렬 안정 역행렬 (상대 ridge)."""
    d = A.shape[0]
    ridge = eps * A.diagonal().mean().clamp(min=1e-12)
    return torch.linalg.inv(A + ridge * torch.eye(d, dtype=A.dtype, device=A.device))


def _normalize_C(C, K):
    """C의 스케일을 K Kᵀ와 맞춰 λ의 의미를 안정화."""
    scale = (K.norm(dim=0) ** 2).mean() / C.diagonal().mean().clamp(min=1e-12)
    return C * scale


def rome_sequential(W, b, K, V_star, C):
    """요청 r을 하나씩: W ← W + (res_r)(C⁻¹k_r)ᵀ / (k_rᵀC⁻¹k_r)."""
    W = W.clone()
    C_inv = _sym_inv(C.double())
    for r in range(K.shape[1]):
        k = K[:, r].double()
        res = V_star[:, r].double() - (W.double() @ k + b.double())
        u = C_inv @ k
        W += (torch.outer(res, u) / (u @ k)).to(W.dtype)
    return W


def memit_batch(W, b, K, V_star, C, lam=1.0):
    """Δ = Res Kᵀ (λC + K Kᵀ)⁻¹ — 보존(λC)과 충족(KKᵀ)의 최소제곱 절충."""
    Kd, Wd = K.double(), W.double()
    Res = V_star.double() - (Wd @ Kd + b.double().unsqueeze(1))
    A = lam * _normalize_C(C.double(), Kd) + Kd @ Kd.T
    delta = Res @ Kd.T @ _sym_inv(A)
    return (Wd + delta).to(W.dtype)


def emmet_batch(W, b, K, V_star, C):
    """등식 제약 해: Δ = Res (Kᵀ C⁻¹ K)⁻¹ (C⁻¹ K)ᵀ — 모든 요청 정확 충족."""
    Kd, Wd = K.double(), W.double()
    Res = V_star.double() - (Wd @ Kd + b.double().unsqueeze(1))
    CK = _sym_inv(C.double()) @ Kd                      # [d_in, R]
    M = _sym_inv(Kd.T @ CK)                             # [R, R]
    delta = Res @ M @ CK.T
    return (Wd + delta).to(W.dtype)


def null_space_projector(K0, rel_threshold=1e-4):
    """보존 키 K₀ [d_in×N]의 null-space 사영 행렬 P (P k₀ ≈ 0)."""
    C0 = K0.double() @ K0.double().T                    # [d_in, d_in]
    eigval, eigvec = torch.linalg.eigh(C0)
    null_mask = eigval < rel_threshold * eigval.max()
    U = eigvec[:, null_mask]
    return U @ U.T                                      # [d_in, d_in]


def alphaedit_batch(W, b, K, V_star, C, K0, lam=1.0, rel_threshold=1e-4):
    """MEMIT식 배치 해를 null-space에서 계산·사영: Δ = Res KpT(λC+KpKpᵀ)⁻¹ 후 ΔP."""
    P = null_space_projector(K0, rel_threshold)
    Kd, Wd = K.double(), W.double()
    Kp = P @ Kd                                         # 보존 공간 성분 제거된 키
    Res = V_star.double() - (Wd @ Kd + b.double().unsqueeze(1))
    A = lam * _normalize_C(C.double(), Kp) + Kp @ Kp.T
    delta = (Res @ Kp.T @ _sym_inv(A)) @ P              # 우측 P: Δ k₀ ≈ 0 보장
    return (Wd + delta).to(W.dtype)


def apply_edit(model, layer, new_W):
    """편집된 W₂를 모델에 반영하고 델타를 반환."""
    fc2 = model.roberta.encoder.layer[layer].output.dense
    delta = (new_W - fc2.weight.data).clone()
    fc2.weight.data.copy_(new_W)
    return delta
