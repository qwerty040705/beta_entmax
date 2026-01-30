# train_new_Synthetic.py
# (NEW: stronger uniformity + leakage + k-gap for STR/k_hat + separation)
import argparse
import os
import time
import numpy as np
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# Utilities
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_prob(p: torch.Tensor, eps_sum: float = 1e-6) -> torch.Tensor:
    """
    - keep exact zeros (sparse heads)
    - nan/inf -> 0, negatives -> 0
    - renorm to sum=1 (if sum=0 -> uniform)
    """
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = p.clamp(min=0.0)
    s = p.sum(dim=-1, keepdim=True)
    m = p.size(-1)
    p = torch.where(s > eps_sum, p / (s + eps_sum), torch.full_like(p, 1.0 / m))
    return p


# ============================================================
# Dataset
# ============================================================

class SyntheticDataset(Dataset):
    def __init__(self, npz_path: str, split: str, split_ratio=(0.8, 0.1, 0.1)):
        data = np.load(npz_path)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        k = data["k"].astype(np.int64)

        N = X.shape[0]
        n_train = int(N * split_ratio[0])
        n_val = int(N * split_ratio[1])

        if split == "train":
            sl = slice(0, n_train)
        elif split == "val":
            sl = slice(n_train, n_train + n_val)
        elif split == "test":
            sl = slice(n_train + n_val, N)
        else:
            raise ValueError("split must be train/val/test")

        self.X = torch.from_numpy(X[sl])
        self.y = torch.from_numpy(y[sl])
        self.k = torch.from_numpy(k[sl])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.k[idx]


# ============================================================
# Models
# ============================================================

class MLPLogits(nn.Module):
    def __init__(self, d, m, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, m),
        )

    def forward(self, x):
        return self.net(x)


class BetaHead(nn.Module):
    def __init__(self, d, hidden=256, beta_min=0.0, beta_max=10.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.beta_min = beta_min
        self.beta_max = beta_max

    def forward(self, x):
        beta = F.softplus(self.net(x)) + self.beta_min
        return torch.clamp(beta, max=self.beta_max)


# ============================================================
# Heads: sparsemax / entmax1.5 / beta-entmax
# ============================================================

class SparsemaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, dim: int = -1):
        z_t = z.transpose(dim, -1)
        orig_shape = z_t.shape
        z2 = z_t.reshape(-1, orig_shape[-1])

        z_max = z2.max(dim=-1, keepdim=True).values
        z2 = z2 - z_max.detach()

        z_sorted, _ = torch.sort(z2, descending=True, dim=-1)
        k = torch.arange(1, z_sorted.size(-1) + 1, device=z.device, dtype=z2.dtype).view(1, -1)
        z_cumsum = torch.cumsum(z_sorted, dim=-1)
        is_gt = (1 + k * z_sorted) > z_cumsum
        k_z = torch.max(k * is_gt.to(z2.dtype), dim=-1).values.clamp(min=1).to(torch.long)

        idx = (k_z - 1).view(-1, 1)
        tau = (z_cumsum.gather(dim=-1, index=idx) - 1.0) / k_z.view(-1, 1).to(z2.dtype)

        p = torch.clamp(z2 - tau, min=0.0)

        supp = (p > 0).to(p.dtype)
        ctx.save_for_backward(supp)
        ctx.dim = dim
        ctx.orig_shape = orig_shape

        return p.reshape(orig_shape).transpose(dim, -1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (supp,) = ctx.saved_tensors
        dim = ctx.dim
        orig_shape = ctx.orig_shape

        g = grad_out.transpose(dim, -1).reshape(-1, orig_shape[-1])
        s = supp
        sum_s = s.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean_sg = (s * g).sum(dim=-1, keepdim=True) / sum_s
        grad_in = s * (g - mean_sg)
        return grad_in.reshape(orig_shape).transpose(dim, -1), None


def sparsemax(z, dim=-1):
    return SparsemaxFunction.apply(z, dim)


class Entmax15Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, dim: int = -1):
        z_t = z.transpose(dim, -1)
        orig_shape = z_t.shape
        z2 = z_t.reshape(-1, orig_shape[-1])

        z_max = z2.max(dim=-1, keepdim=True).values
        z2 = z2 - z_max.detach()

        z_sorted, _ = torch.sort(z2, descending=True, dim=-1)
        S1 = torch.cumsum(z_sorted, dim=-1)
        S2 = torch.cumsum(z_sorted * z_sorted, dim=-1)
        k = torch.arange(1, z_sorted.size(-1) + 1, device=z.device, dtype=z2.dtype).view(1, -1)

        disc = S1 * S1 - k * (S2 - 1.0)
        disc = torch.clamp(disc, min=0.0)
        tau = (S1 - torch.sqrt(disc)) / k

        support = tau < z_sorted
        k_z = support.sum(dim=-1).clamp(min=1)
        idx = (k_z - 1).view(-1, 1)
        tau_star = tau.gather(dim=-1, index=idx)

        p_unn = torch.clamp(z2 - tau_star, min=0.0) ** 2
        p = p_unn / (p_unn.sum(dim=-1, keepdim=True) + 1e-6)

        ctx.save_for_backward(p)
        ctx.dim = dim
        ctx.orig_shape = orig_shape
        return p.reshape(orig_shape).transpose(dim, -1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (p,) = ctx.saved_tensors
        dim = ctx.dim
        orig_shape = ctx.orig_shape

        g = grad_out.transpose(dim, -1).reshape(-1, orig_shape[-1])

        supp = p > 0
        s = torch.zeros_like(p)
        s[supp] = torch.sqrt(p[supp].clamp(min=0.0))

        sum_s = s.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        mean_sg = (s * g).sum(dim=-1, keepdim=True) / sum_s
        grad_in = s * (g - mean_sg)

        return grad_in.reshape(orig_shape).transpose(dim, -1), None


def entmax15(z, dim=-1):
    return Entmax15Function.apply(z, dim)


class BetaEntmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, beta: torch.Tensor, alpha: float = 1.5, dim: int = -1, n_iter: int = 30):
        z_t = z.transpose(dim, -1)
        orig_shape = z_t.shape
        z2 = z_t.reshape(-1, orig_shape[-1])
        beta2 = beta.reshape(-1, 1).to(z2.dtype)

        z_max = z2.max(dim=-1, keepdim=True).values
        z2s = z2 - z_max.detach()

        r = 1.0 / (alpha - 1.0)
        eps_beta = 1e-7

        tau_high = z2s.max(dim=-1, keepdim=True).values
        tau_low = tau_high - 2.0

        def compute_p_and_aux(tau):
            diff = z2s - tau
            pos = diff > 0

            inner = (r ** 2) + 4.0 * beta2 * diff
            inner = torch.clamp(inner, min=0.0)
            sqrt_term = torch.sqrt(inner)

            t = (-r + sqrt_term) / (2.0 * beta2 + 1e-6)
            p_beta_unn = torch.pow(torch.clamp(t, min=0.0), r)

            u = torch.clamp((alpha - 1.0) * diff, min=0.0)
            p_ent_unn = torch.pow(u, r)

            p_unn = torch.where(beta2 > eps_beta, p_beta_unn, p_ent_unn)
            p_unn = p_unn * pos.to(p_unn.dtype)
            p = p_unn / (p_unn.sum(dim=-1, keepdim=True) + 1e-6)
            return p, p_unn, t, sqrt_term, diff

        # expand tau_low until sum >= 1
        for _ in range(30):
            _, p_unn_tmp, *_ = compute_p_and_aux(tau_low)
            if torch.all(p_unn_tmp.sum(dim=-1, keepdim=True) >= 1.0):
                break
            tau_low = tau_low - 2.0

        for _ in range(n_iter):
            tau_mid = 0.5 * (tau_low + tau_high)
            _, p_unn_mid, *_ = compute_p_and_aux(tau_mid)
            s_mid = p_unn_mid.sum(dim=-1, keepdim=True)
            tau_low = torch.where(s_mid > 1.0, tau_mid, tau_low)
            tau_high = torch.where(s_mid <= 1.0, tau_mid, tau_high)

        p, p_unn, t, sqrt_term, diff = compute_p_and_aux(tau_high)

        ctx.save_for_backward(p, p_unn, t, sqrt_term, diff, beta2)
        ctx.alpha = alpha
        ctx.dim = dim
        ctx.orig_shape = orig_shape
        return p.reshape(orig_shape).transpose(dim, -1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        p, p_unn, t, sqrt_term, diff, beta2 = ctx.saved_tensors
        alpha = ctx.alpha
        dim = ctx.dim
        orig_shape = ctx.orig_shape

        g = grad_out.transpose(dim, -1).reshape(-1, orig_shape[-1])

        r = 1.0 / (alpha - 1.0)
        eps_beta = 1e-7

        supp = p_unn > 0
        sqrt_safe = sqrt_term.clamp(min=1e-6)
        t_safe = t.clamp(min=1e-6)

        # Jacobian weights
        w_beta = torch.zeros_like(p_unn)
        w_beta[supp] = r * torch.pow(t_safe[supp], r - 1.0) / sqrt_safe[supp]

        w_ent = torch.zeros_like(p_unn)
        w_ent[supp] = torch.pow(p[supp].clamp(min=0.0), 2.0 - alpha)

        w = torch.where(beta2 > eps_beta, w_beta, w_ent)
        w = w * supp.to(w.dtype)

        sum_w = w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        gw = (g * w).sum(dim=-1, keepdim=True)
        grad_z = w * (g - gw / sum_w)

        grad_beta = None
        if ctx.needs_input_grad[1]:
            beta_safe = beta2.clamp(min=1e-6)
            s = sqrt_safe
            dt_db = (beta_safe * diff / s + 0.5 * (r - s)) / (beta_safe ** 2)
            dh_dbeta_beta = r * torch.pow(t_safe, r - 1.0) * dt_db

            dh_dbeta = torch.where(beta2 > eps_beta, dh_dbeta_beta, torch.zeros_like(dh_dbeta_beta))
            dh_dbeta = dh_dbeta * supp.to(dh_dbeta.dtype)

            sum_dh = dh_dbeta.sum(dim=-1, keepdim=True)
            dp_dbeta = dh_dbeta - w * (sum_dh / sum_w)
            grad_beta = (g * dp_dbeta).sum(dim=-1, keepdim=True)

        return grad_z.reshape(orig_shape).transpose(dim, -1), grad_beta, None, None, None


def beta_entmax(z: torch.Tensor, beta: torch.Tensor, alpha: float = 1.5, dim: int = -1, n_iter: int = 30):
    return BetaEntmaxFunction.apply(z, beta, alpha, dim, n_iter)


# ============================================================
# New Losses (핵심)
# ============================================================

def multilabel_distribution(y: torch.Tensor) -> torch.Tensor:
    k = y.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return y / k


def loss_task_ce(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    q = y/k,  L = -sum q log p
    """
    q = multilabel_distribution(y)
    p = safe_prob(p)
    return -(q * torch.log(p.clamp(min=eps))).sum(dim=-1).mean()


def loss_mass_leakage(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    leakage = 1 - sum_{i in S} p_i  (= sum_{j notin S} p_j)
    이걸 강하게 줄이면 FP가 확 줄고, thresholding이 쉬워짐.
    """
    p = safe_prob(p)
    mask = (y > 0.5).to(p.dtype)
    s_mass = (p * mask).sum(dim=-1)  # [B]
    leakage = (1.0 - s_mass).clamp(min=0.0)
    return (leakage ** 2).mean()


def loss_uniform_l2_on_S(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    네가 원한 '더더더 uniform'을 KL보다 더 세게:
    1) S에서 확률 질량을 renorm 해서 p_tilde 만들고
    2) p_tilde가 1/k로 수렴하도록 L2로 압박
    """
    p = safe_prob(p)
    mask = (y > 0.5).to(p.dtype)
    k = mask.sum(dim=-1).clamp(min=1.0)  # [B]

    pS = p * mask
    s_mass = pS.sum(dim=-1, keepdim=True).clamp(min=eps)
    p_tilde = pS / s_mass  # S에서만 합 1

    target = (1.0 / k).unsqueeze(-1)  # [B,1]
    l2 = ((p_tilde - target) ** 2) * mask
    # 평균을 k로 나눠서 k크기 변화에 민감하지 않게
    return (l2.sum(dim=-1) / k).mean()


def loss_separation_posneg(p: torch.Tensor, y: torch.Tensor, margin: float = 0.01) -> torch.Tensor:
    """
    STR을 직접 키우는 hinge:
      min_{i in S} p_i >= max_{j not in S} p_j + margin
    """
    p = safe_prob(p)
    mask = (y > 0.5)

    pos_min = p.masked_fill(~mask, float("inf")).min(dim=-1).values
    neg_max = p.masked_fill(mask, float("-inf")).max(dim=-1).values

    # k==m이면 neg_max=-inf -> 0으로 처리
    neg_max = torch.where(torch.isfinite(neg_max), neg_max, torch.zeros_like(neg_max))
    # 안정성
    pos_min = torch.where(torch.isfinite(pos_min), pos_min, torch.zeros_like(pos_min))

    return F.relu(margin + neg_max - pos_min).mean()


def loss_kgap_sorted(p: torch.Tensor, k_true: torch.Tensor, margin_k: float = 0.02) -> torch.Tensor:
    """
    k_hat/STR을 동시에 올리는 핵심:
      p_(k) - p_(k+1) 를 크게 만들면
      theta가 그 사이에 들어갈 수 있는 구간(=STR surrogate)이 커짐.
    """
    p = safe_prob(p)
    B, m = p.shape
    k = k_true.clamp(min=1, max=m).to(torch.long)  # [B]

    p_sorted, _ = torch.sort(p, dim=-1, descending=True)

    idx_k = (k - 1).view(B, 1)  # 0-index
    p_k = p_sorted.gather(1, idx_k).squeeze(1)

    # k==m 이면 p_(k+1)=0으로 둠
    idx_k1 = k.view(B, 1).clamp(max=m - 1)
    p_k1 = p_sorted.gather(1, idx_k1).squeeze(1)
    p_k1 = torch.where(k == m, torch.zeros_like(p_k1), p_k1)

    gap = p_k - p_k1
    return F.relu(margin_k - gap).mean()


def beta_reg_loss(beta: torch.Tensor) -> torch.Tensor:
    return (beta ** 2).mean()


# ============================================================
# Metrics (val에서 보고용)
# ============================================================

@torch.no_grad()
def compute_metrics_from_loader(model_g, beta_head, loader,
                                head: str, fixed_beta: float,
                                theta_min: float, theta_max: float, theta_points: int,
                                device):
    model_g.eval()
    if beta_head is not None:
        beta_head.eval()

    all_p = []
    all_y = []
    all_k = []

    for x, y, k in loader:
        x = x.to(device)
        y = y.to(device)
        k = k.to(device)

        z = model_g(x)

        if head == "softmax":
            p = F.softmax(z, dim=-1)
        elif head == "sparsemax":
            p = sparsemax(z, dim=-1)
        elif head == "entmax15":
            p = entmax15(z, dim=-1)
        elif head == "beta_entmax":
            if beta_head is None:
                beta = torch.full((x.size(0), 1), float(fixed_beta), device=device, dtype=z.dtype)
            else:
                beta = beta_head(x)
            p = beta_entmax(z, beta, alpha=1.5, dim=-1, n_iter=30)
        else:
            raise ValueError(head)

        p = safe_prob(p)
        all_p.append(p.detach().cpu())
        all_y.append(y.detach().cpu())
        all_k.append(k.detach().cpu())

    if not all_p:
        return {}

    P = torch.cat(all_p, dim=0)  # [N,m]
    Y = torch.cat(all_y, dim=0)  # [N,m]
    K = torch.cat(all_k, dim=0)  # [N]

    mask = (Y > 0.5)

    # STR proxy: interval length where threshold recovers exact support
    pos_min = P.masked_fill(~mask, float("inf")).min(dim=-1).values
    neg_max = P.masked_fill(mask, float("-inf")).max(dim=-1).values
    neg_max = torch.where(torch.isfinite(neg_max), neg_max, torch.zeros_like(neg_max))
    pos_min = torch.where(torch.isfinite(pos_min), pos_min, torch.zeros_like(pos_min))
    STR = (pos_min - neg_max).clamp(min=0.0).mean().item()

    # Uniformity (on S, renorm)
    kf = mask.sum(dim=-1).clamp(min=1).to(P.dtype)
    pS = P * mask.to(P.dtype)
    s_mass = pS.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    p_tilde = pS / s_mass
    target = (1.0 / kf).unsqueeze(-1)
    U_L2 = ((((p_tilde - target) ** 2) * mask.to(P.dtype)).sum(dim=-1) / kf).mean().item()

    # Leakage
    leakage = (1.0 - pS.sum(dim=-1)).clamp(min=0.0)
    LEAK = (leakage.mean().item())

    # F1 sweep: choose best theta (micro-F1)
    tmin = max(theta_min, 1e-8)
    tmax = max(theta_max, tmin * 10.0)
    thetas = torch.logspace(
        np.log10(tmin),
        np.log10(tmax),
        steps=int(theta_points),
        device="cpu",
        dtype=P.dtype
    )

    best = {"f1": -1.0, "theta": None, "kacc": None, "kmae": None}
    for th in thetas:
        thv = float(th.item())
        pred = (P >= thv)

        TP = (pred & mask).sum().item()
        FP = (pred & ~mask).sum().item()
        FN = ((~pred) & mask).sum().item()

        denom = (2 * TP + FP + FN)
        f1 = (2 * TP / denom) if denom > 0 else 0.0

        khat = pred.sum(dim=-1).to(torch.long)
        kacc = (khat == K).float().mean().item()
        kmae = (khat.to(torch.float32) - K.to(torch.float32)).abs().mean().item()

        if f1 > best["f1"]:
            best = {"f1": f1, "theta": thv, "kacc": kacc, "kmae": kmae}

    return {
        "STR": STR,
        "U_L2": U_L2,
        "Leak": LEAK,
        "F1_best": best["f1"],
        "theta_best": best["theta"],
        "KAcc_best": best["kacc"],
        "KMAE_best": best["kmae"],
    }


# ============================================================
# Train / Eval (NEW objective)
# ============================================================

def forward_head(model_g, beta_head, x, head: str, fixed_beta: float, device):
    z = model_g(x)
    beta = None

    if head == "softmax":
        p = F.softmax(z, dim=-1)
    elif head == "sparsemax":
        p = sparsemax(z, dim=-1)
    elif head == "entmax15":
        p = entmax15(z, dim=-1)
    elif head == "beta_entmax":
        if beta_head is None:
            beta = torch.full((x.size(0), 1), float(fixed_beta), device=device, dtype=z.dtype)
        else:
            beta = beta_head(x)
        p = beta_entmax(z, beta, alpha=1.5, dim=-1, n_iter=30)
    else:
        raise ValueError(head)

    return safe_prob(p), beta


@torch.no_grad()
def evaluate_new(model_g, beta_head, loader,
                 head: str, fixed_beta: float,
                 # loss weights
                 w_uni: float, w_leak: float, w_sep: float, sep_margin: float,
                 w_kgap: float, kgap_margin: float,
                 eta_beta: float,
                 device):
    model_g.eval()
    if beta_head is not None:
        beta_head.eval()

    losses = []
    for x, y, k in loader:
        x = x.to(device)
        y = y.to(device)
        k = k.to(device)

        p, beta = forward_head(model_g, beta_head, x, head=head, fixed_beta=fixed_beta, device=device)

        loss = loss_task_ce(p, y)

        if w_leak > 0:
            loss = loss + w_leak * loss_mass_leakage(p, y)
        if w_sep > 0:
            loss = loss + w_sep * loss_separation_posneg(p, y, margin=sep_margin)
        if w_kgap > 0:
            loss = loss + w_kgap * loss_kgap_sorted(p, k, margin_k=kgap_margin)
        if w_uni > 0:
            loss = loss + w_uni * loss_uniform_l2_on_S(p, y)

        if head == "beta_entmax" and (beta_head is not None) and (eta_beta > 0.0):
            loss = loss + eta_beta * beta_reg_loss(beta)

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else 0.0


def train_one_epoch_new(model_g, beta_head, opt, loader,
                        head: str, fixed_beta: float,
                        # loss weights
                        w_uni: float, w_leak: float, w_sep: float, sep_margin: float,
                        w_kgap: float, kgap_margin: float,
                        eta_beta: float,
                        device, grad_clip: float):
    model_g.train()
    if beta_head is not None:
        beta_head.train()

    losses = []
    t0 = time.time()

    pbar = tqdm(loader, desc="Train", leave=False)
    for it, (x, y, k) in enumerate(pbar, start=1):
        x = x.to(device)
        y = y.to(device)
        k = k.to(device)

        p, beta = forward_head(model_g, beta_head, x, head=head, fixed_beta=fixed_beta, device=device)

        loss = loss_task_ce(p, y)

        if w_leak > 0:
            loss = loss + w_leak * loss_mass_leakage(p, y)
        if w_sep > 0:
            loss = loss + w_sep * loss_separation_posneg(p, y, margin=sep_margin)
        if w_kgap > 0:
            loss = loss + w_kgap * loss_kgap_sorted(p, k, margin_k=kgap_margin)
        if w_uni > 0:
            loss = loss + w_uni * loss_uniform_l2_on_S(p, y)

        if head == "beta_entmax" and (beta_head is not None) and (eta_beta > 0.0):
            loss = loss + eta_beta * beta_reg_loss(beta)

        if not torch.isfinite(loss):
            print("[WARN] Non-finite loss detected. Skipping batch.")
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            params = list(model_g.parameters())
            if beta_head is not None:
                params += list(beta_head.parameters())
            torch.nn.utils.clip_grad_norm_(params, grad_clip)

        opt.step()

        losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))

        # rough ETA within epoch
        elapsed = time.time() - t0
        rate = elapsed / max(it, 1)
        eta = rate * (len(loader) - it)

        pbar.set_postfix(loss=f"{mean_loss:.5f}", eta=f"{eta:,.1f}s")

    return float(np.mean(losses)) if losses else 0.0


# ============================================================
# Main
# ============================================================

@dataclass
class TrainArgs:
    data: str
    out: str
    head: str
    epochs: int
    batch: int
    lr: float
    wd: float
    hidden: int
    seed: int

    fixed_beta: float
    train_beta_head: int
    beta_max: float

    # NEW objective weights
    w_uni: float
    w_leak: float
    w_sep: float
    sep_margin: float
    w_kgap: float
    kgap_margin: float

    eta_beta: float
    grad_clip: float
    num_workers: int

    # val sweep range (for reporting)
    val_theta_min: float
    val_theta_max: float
    val_theta_points: int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="ckpt_new.pt")
    ap.add_argument("--head", type=str, required=True, choices=["softmax", "sparsemax", "entmax15", "beta_entmax"])

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--fixed_beta", type=float, default=1.0)
    ap.add_argument("--train_beta_head", type=int, default=1)
    ap.add_argument("--beta_max", type=float, default=10.0)

    # NEW objective weights (이게 핵심)
    ap.add_argument("--w_uni", type=float, default=0.5)         # uniform L2 on S (더 세게)
    ap.add_argument("--w_leak", type=float, default=2.0)        # leakage 제거 (FP 억제 핵심)
    ap.add_argument("--w_sep", type=float, default=1.0)         # pos/neg separation -> STR
    ap.add_argument("--sep_margin", type=float, default=0.01)
    ap.add_argument("--w_kgap", type=float, default=1.0)        # p_k - p_{k+1} gap -> STR/k_hat
    ap.add_argument("--kgap_margin", type=float, default=0.02)

    ap.add_argument("--eta_beta", type=float, default=1e-3)     # beta blow-up 방지
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=0)

    # val reporting sweep range
    ap.add_argument("--val_theta_min", type=float, default=1e-6)
    ap.add_argument("--val_theta_max", type=float, default=1e-2)
    ap.add_argument("--val_theta_points", type=int, default=16)

    args = ap.parse_args()

    targs = TrainArgs(
        data=args.data,
        out=args.out,
        head=args.head,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        wd=args.wd,
        hidden=args.hidden,
        seed=args.seed,

        fixed_beta=args.fixed_beta,
        train_beta_head=args.train_beta_head,
        beta_max=args.beta_max,

        w_uni=args.w_uni,
        w_leak=args.w_leak,
        w_sep=args.w_sep,
        sep_margin=args.sep_margin,
        w_kgap=args.w_kgap,
        kgap_margin=args.kgap_margin,

        eta_beta=args.eta_beta,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,

        val_theta_min=args.val_theta_min,
        val_theta_max=args.val_theta_max,
        val_theta_points=args.val_theta_points,
    )

    set_seed(targs.seed)
    device = get_device()
    print(f"[Device] {device}")
    print(f"[Train-New] data={targs.data}, head={targs.head}, out={targs.out}")
    print(f"[Weights] w_uni={targs.w_uni}, w_leak={targs.w_leak}, w_sep={targs.w_sep}, w_kgap={targs.w_kgap}")
    print(f"[Margins] sep_margin={targs.sep_margin}, kgap_margin={targs.kgap_margin}")
    if targs.head == "beta_entmax":
        print(f"[Beta] train_beta_head={targs.train_beta_head}, fixed_beta={targs.fixed_beta}, beta_max={targs.beta_max}, eta_beta={targs.eta_beta}")

    data_npz = np.load(targs.data)
    d = int(data_npz["d"])
    m = int(data_npz["m"])

    ds_train = SyntheticDataset(targs.data, "train")
    ds_val = SyntheticDataset(targs.data, "val")
    train_loader = DataLoader(ds_train, batch_size=targs.batch, shuffle=True, num_workers=targs.num_workers)
    val_loader = DataLoader(ds_val, batch_size=targs.batch, shuffle=False, num_workers=targs.num_workers)

    model_g = MLPLogits(d=d, m=m, hidden=targs.hidden).to(device)

    beta_head = None
    if targs.head == "beta_entmax" and int(targs.train_beta_head) == 1:
        beta_head = BetaHead(d=d, hidden=256, beta_min=0.0, beta_max=float(targs.beta_max)).to(device)

    params = list(model_g.parameters())
    if beta_head is not None:
        params += list(beta_head.parameters())
    opt = torch.optim.AdamW(params, lr=targs.lr, weight_decay=targs.wd)

    # 모델 선택 기준:
    # - val loss만 보면 (uniform/leak/sep/kgap) tradeoff가 섞여서 애매할 수 있음
    # - 그래서 "F1_best + STR + KAcc"를 섞은 score로 best를 잡는 게 더 안정적
    def score_from_metrics(mt: dict):
        if not mt:
            return -1e9
        # 가중치는 상황에 맞게 바꿔도 됨
        return (1.0 * mt["F1_best"] + 0.5 * mt["STR"] + 0.5 * mt["KAcc_best"] - 0.1 * mt["Leak"])

    best_score = -1e9
    best_state = None

    epoch_times = []
    for epoch in range(1, targs.epochs + 1):
        ep_t0 = time.time()

        tr_loss = train_one_epoch_new(
            model_g=model_g,
            beta_head=beta_head,
            opt=opt,
            loader=train_loader,
            head=targs.head,
            fixed_beta=targs.fixed_beta,
            w_uni=targs.w_uni,
            w_leak=targs.w_leak,
            w_sep=targs.w_sep,
            sep_margin=targs.sep_margin,
            w_kgap=targs.w_kgap,
            kgap_margin=targs.kgap_margin,
            eta_beta=targs.eta_beta,
            device=device,
            grad_clip=targs.grad_clip,
        )

        va_loss = evaluate_new(
            model_g=model_g,
            beta_head=beta_head,
            loader=val_loader,
            head=targs.head,
            fixed_beta=targs.fixed_beta,
            w_uni=targs.w_uni,
            w_leak=targs.w_leak,
            w_sep=targs.w_sep,
            sep_margin=targs.sep_margin,
            w_kgap=targs.w_kgap,
            kgap_margin=targs.kgap_margin,
            eta_beta=targs.eta_beta,
            device=device,
        )

        metrics = compute_metrics_from_loader(
            model_g=model_g,
            beta_head=beta_head,
            loader=val_loader,
            head=targs.head,
            fixed_beta=targs.fixed_beta,
            theta_min=targs.val_theta_min,
            theta_max=targs.val_theta_max,
            theta_points=targs.val_theta_points,
            device=device,
        )

        ep_time = time.time() - ep_t0
        epoch_times.append(ep_time)
        avg_ep = sum(epoch_times) / len(epoch_times)
        eta_total = avg_ep * (targs.epochs - epoch)

        msg = (
            f"[Epoch {epoch:03d}] train_loss={tr_loss:.6f} val_loss={va_loss:.6f} | "
            f"F1*={metrics.get('F1_best',0):.4f} KAcc*={metrics.get('KAcc_best',0):.4f} "
            f"KMAE*={metrics.get('KMAE_best',0):.3f} STR={metrics.get('STR',0):.4f} "
            f"Leak={metrics.get('Leak',0):.4f} U_L2={metrics.get('U_L2',0):.4f} "
            f"theta*={metrics.get('theta_best',0):.2e} | ETA_total={eta_total:,.1f}s"
        )
        print(msg)

        sc = score_from_metrics(metrics)
        if sc > best_score:
            best_score = sc
            best_state = {
                "model_g": model_g.state_dict(),
                "beta_head": None if beta_head is None else beta_head.state_dict(),
                "args": asdict(targs),
                "best_score": best_score,
                "best_metrics": metrics,
            }

    if best_state is None:
        best_state = {
            "model_g": model_g.state_dict(),
            "beta_head": None if beta_head is None else beta_head.state_dict(),
            "args": asdict(targs),
            "best_score": best_score,
            "best_metrics": {},
        }

    os.makedirs(os.path.dirname(targs.out) or ".", exist_ok=True)
    torch.save(best_state, targs.out)
    print(f"[Saved] {targs.out}  (best_score={best_score:.6f})")
    if "best_metrics" in best_state:
        print("[Best metrics]", best_state["best_metrics"])


if __name__ == "__main__":
    main()


"""
# ============================================================
# Example commands
# ============================================================

# MAIN (네가 원하는 방향: card_loss 없이도 k_hat/STR/F1/uniform을 같이 올리는 NEW objective)
python3 train_new_Synthetic.py --data ./Synthetic.npz --out ckpt_beta_new.pt --head beta_entmax --epochs 20 \
  --train_beta_head 1 --beta_max 10.0 \
  --w_uni 0.5 --w_leak 2.0 --w_sep 1.0 --sep_margin 0.01 --w_kgap 1.0 --kgap_margin 0.02 \
  --eta_beta 1e-3 \
  --val_theta_min 1e-6 --val_theta_max 1e-2 --val_theta_points 16

# 만약 uniform을 더 "과격하게" (nonzero를 진짜 1/k로 몰기) 하고 싶으면:
#   --w_uni 1.0 ~ 3.0
# 대신 leak/sep/kgap가 같이 있어야 FP가 폭발 안 함.
"""
