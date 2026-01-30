import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import Counter
from typing import Tuple, Optional, Dict


# ============================================================
# Device / safety
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_prob_eval(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = p.clamp(min=0.0)
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
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
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.beta_min = beta_min
        self.beta_max = beta_max

    def forward(self, x):
        beta = F.softplus(self.net(x)) + self.beta_min
        return torch.clamp(beta, max=self.beta_max)


# ============================================================
# Heads (same as train)
# ============================================================

def softmax_head(z):
    return F.softmax(z, dim=-1)


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

        out = p.reshape(orig_shape).transpose(dim, -1)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (supp,) = ctx.saved_tensors
        dim = ctx.dim
        orig_shape = ctx.orig_shape

        g_t = grad_out.transpose(dim, -1).reshape(-1, orig_shape[-1])
        s = supp
        sum_s = s.sum(dim=-1, keepdim=True).clamp(min=1.0)
        sg = s * g_t
        mean_sg = sg.sum(dim=-1, keepdim=True) / sum_s
        grad_in = s * (g_t - mean_sg)

        grad_in = grad_in.reshape(orig_shape).transpose(dim, -1)
        return grad_in, None


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
        zs_cum = torch.cumsum(z_sorted, dim=-1)
        z2s_cum = torch.cumsum(z_sorted * z_sorted, dim=-1)
        k = torch.arange(1, z_sorted.size(-1) + 1, device=z.device, dtype=z2.dtype).view(1, -1)

        disc = zs_cum * zs_cum - k * (z2s_cum - 1.0)
        disc = torch.clamp(disc, min=0.0)
        tau = (zs_cum - torch.sqrt(disc)) / k

        support = tau < z_sorted
        k_z = support.sum(dim=-1).clamp(min=1)

        idx = (k_z - 1).view(-1, 1)
        tau_star = tau.gather(dim=-1, index=idx)

        p_unn = torch.clamp(z2 - tau_star, min=0.0) ** 2
        p = p_unn / (p_unn.sum(dim=-1, keepdim=True) + 1e-12)

        ctx.save_for_backward(p)
        ctx.dim = dim
        ctx.orig_shape = orig_shape

        out = p.reshape(orig_shape).transpose(dim, -1)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (p,) = ctx.saved_tensors
        dim = ctx.dim
        orig_shape = ctx.orig_shape

        g_t = grad_out.transpose(dim, -1).reshape(-1, orig_shape[-1])

        supp = p > 0
        s = torch.zeros_like(p)
        s[supp] = torch.sqrt(p[supp].clamp(min=0.0))

        sum_s = s.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        sg = s * g_t
        mean_sg = sg.sum(dim=-1, keepdim=True) / sum_s
        grad_in = s * (g_t - mean_sg)

        grad_in = grad_in.reshape(orig_shape).transpose(dim, -1)
        return grad_in, None


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

            t = (-r + sqrt_term) / (2.0 * beta2 + 1e-12)
            p_beta_unn = torch.pow(torch.clamp(t, min=0.0), r)

            u = torch.clamp((alpha - 1.0) * diff, min=0.0)
            p_ent_unn = torch.pow(u, r)

            p_unn = torch.where(beta2 > eps_beta, p_beta_unn, p_ent_unn)
            p_unn = p_unn * pos.to(p_unn.dtype)

            p = p_unn / (p_unn.sum(dim=-1, keepdim=True) + 1e-12)
            return p, p_unn, t, sqrt_term, diff

        for _ in range(20):
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

        out = p.reshape(orig_shape).transpose(dim, -1)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        p, p_unn, t, sqrt_term, diff, beta2 = ctx.saved_tensors
        alpha = ctx.alpha
        dim = ctx.dim
        orig_shape = ctx.orig_shape

        g_t = grad_out.transpose(dim, -1).reshape(-1, orig_shape[-1])

        r = 1.0 / (alpha - 1.0)
        eps_beta = 1e-7

        supp = p_unn > 0
        sqrt_safe = sqrt_term.clamp(min=1e-12)
        t_safe = t.clamp(min=1e-12)

        w_beta = torch.zeros_like(p_unn)
        w_beta[supp] = r * torch.pow(t_safe[supp], r - 1.0) / sqrt_safe[supp]

        w_ent = torch.zeros_like(p_unn)
        w_ent[supp] = torch.pow(p[supp].clamp(min=0.0), 2.0 - alpha)

        w = torch.where(beta2 > eps_beta, w_beta, w_ent)
        w = w * supp.to(w.dtype)

        sum_w = w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        gw = (g_t * w).sum(dim=-1, keepdim=True)
        grad_z = w * (g_t - gw / sum_w)

        grad_beta = None
        if ctx.needs_input_grad[1]:
            beta_safe = beta2.clamp(min=1e-12)
            s = sqrt_safe
            dt_db = (beta_safe * diff / s + 0.5 * (r - s)) / (beta_safe ** 2)
            dh_dbeta_beta = r * torch.pow(t_safe, r - 1.0) * dt_db

            dh_dbeta = torch.where(beta2 > eps_beta, dh_dbeta_beta, torch.zeros_like(dh_dbeta_beta))
            dh_dbeta = dh_dbeta * supp.to(dh_dbeta.dtype)

            sum_dh = dh_dbeta.sum(dim=-1, keepdim=True)
            dp_dbeta = dh_dbeta - w * (sum_dh / sum_w)
            grad_beta = (g_t * dp_dbeta).sum(dim=-1, keepdim=True)

        grad_z = grad_z.reshape(orig_shape).transpose(dim, -1)
        return grad_z, grad_beta, None, None, None


def beta_entmax(z, beta, alpha: float = 1.5, dim: int = -1, n_iter: int = 30):
    return BetaEntmaxFunction.apply(z, beta, alpha, dim, n_iter)


# ============================================================
# Metrics: coverage + uniformity on TRUE support
# ============================================================

@torch.no_grad()
def compute_support_mass_stats(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-12):
    """
    True-support mass:
      s_mass = sum_{i in S} p_i,  S = {i: y_i=1}
    and coverage NLL:
      cov_nll = E[ -log(s_mass + eps) ]
    Returns: (mean_s_mass, cov_nll)
    """
    mask = y > 0.5
    pS = p * mask.to(p.dtype)
    s_mass = pS.sum(dim=1)  # [B]
    mean_s = float(s_mass.mean().cpu())
    cov_nll = float((-torch.log(s_mass.clamp(min=eps))).mean().cpu())
    return mean_s, cov_nll


@torch.no_grad()
def compute_U_gap_oracle(p: torch.Tensor, y: torch.Tensor) -> float:
    """
    Oracle uniformity on TRUE support:
      U_gap = k * (max_{i in S} p_i - min_{i in S} p_i)
    NOTE: This can be small even when s_mass is tiny, so report with support-mass metrics.
    """
    mask = y > 0.5
    k = mask.sum(dim=1).clamp(min=1).to(p.dtype)

    p_support = p.masked_fill(~mask, float("-inf"))
    p_max = p_support.max(dim=1).values

    p_support2 = p.masked_fill(~mask, float("inf"))
    p_min = p_support2.min(dim=1).values

    ugap = k * (p_max - p_min)
    ugap = torch.where(torch.isfinite(ugap), ugap, torch.zeros_like(ugap))
    return float(ugap.mean().cpu())


@torch.no_grad()
def compute_U_KL_oracle(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Oracle uniformity on TRUE support:
      U_KL = KL( p_tilde_S || Uniform(S) )
      p_tilde_S = p restricted to S, renormalized on S.
    NOTE: Can be misleading if s_mass is tiny -> use mass-weighted variants too.
    """
    mask = y > 0.5
    k = mask.sum(dim=1).clamp(min=1).to(p.dtype)

    pS = p * mask.to(p.dtype)
    s_mass = pS.sum(dim=1, keepdim=True).clamp(min=eps)

    p_tilde = (pS / s_mass).clamp(min=eps)
    log_p = torch.log(p_tilde)

    log_u = -torch.log(k).unsqueeze(1)  # log(1/k)
    kl = (p_tilde * (log_p - log_u) * mask.to(p.dtype)).sum(dim=1)
    kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
    return float(kl.mean().cpu())


@torch.no_grad()
def compute_U_gap_weighted(p: torch.Tensor, y: torch.Tensor) -> float:
    """
    Mass-weighted oracle U_gap:
      U_gap^w = E[ s_mass * k * (max_S p - min_S p) ]
    where s_mass = sum_{i in S} p_i
    """
    mask = y > 0.5
    k = mask.sum(dim=1).clamp(min=1).to(p.dtype)

    pS = p * mask.to(p.dtype)
    s_mass = pS.sum(dim=1).clamp(min=0.0)  # [B]

    p_support = p.masked_fill(~mask, float("-inf"))
    p_max = p_support.max(dim=1).values
    p_support2 = p.masked_fill(~mask, float("inf"))
    p_min = p_support2.min(dim=1).values

    ugap = k * (p_max - p_min)
    ugap = torch.where(torch.isfinite(ugap), ugap, torch.zeros_like(ugap))

    out = (s_mass * ugap).mean()
    return float(out.cpu())


@torch.no_grad()
def compute_U_KL_weighted(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Mass-weighted oracle U_KL:
      U_KL^w = E[ s_mass * KL(p_tilde_S || Uniform(S)) ]
    """
    mask = y > 0.5
    k = mask.sum(dim=1).clamp(min=1).to(p.dtype)

    pS = p * mask.to(p.dtype)
    s_mass = pS.sum(dim=1, keepdim=True).clamp(min=eps)

    p_tilde = (pS / s_mass).clamp(min=eps)
    log_p = torch.log(p_tilde)
    log_u = -torch.log(k).unsqueeze(1)

    kl = (p_tilde * (log_p - log_u) * mask.to(p.dtype)).sum(dim=1)  # [B]
    kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))

    w = s_mass.squeeze(1).clamp(min=0.0)  # [B]
    out = (w * kl).mean()
    return float(out.cpu())


# ============================================================
# Pred-support uniformity at theta*
# ============================================================

def pred_support_uniformity(p: np.ndarray, theta: float, eps: float = 1e-12):
    """
    Predicted-support uniformity at threshold theta.
    Let S_hat = {i : p_i >= theta}, k_hat = |S_hat|.
    Compute:
      - pred_mass = sum_{i in S_hat} p_i
      - pred_U_KL = KL( p_{S_hat} / pred_mass || Uniform(S_hat) )   (0 if k_hat<=1)
      - pred_U_gap = k_hat * (max_{S_hat} p - min_{S_hat} p)        (0 if k_hat<=1)
    Returns mean over samples:
      (mean_k_hat, mean_pred_mass, mean_pred_U_KL, mean_pred_U_gap)
    """
    P = p
    mask = (P >= float(theta))  # [N,m]
    k_hat = mask.sum(axis=1).astype(np.int64)  # [N]
    pred_mass = (P * mask).sum(axis=1)  # [N]

    pred_U_KL = np.zeros(P.shape[0], dtype=np.float64)
    pred_U_gap = np.zeros(P.shape[0], dtype=np.float64)

    for n in range(P.shape[0]):
        k = int(k_hat[n])
        if k <= 1:
            continue
        idx = mask[n]
        mass = max(float(pred_mass[n]), eps)
        ptilde = (P[n, idx] / mass).clip(min=eps)
        pred_U_KL[n] = float(np.sum(ptilde * (np.log(ptilde) - np.log(1.0 / k))))
        pred_U_gap[n] = float(k * (P[n, idx].max() - P[n, idx].min()))

    return (
        float(k_hat.mean()),
        float(pred_mass.mean()),
        float(pred_U_KL.mean()),
        float(pred_U_gap.mean()),
    )


# ============================================================
# Theta grid helpers
# ============================================================

def make_theta_grid(theta_min: float, theta_max: float, theta_points: int, logspace: bool):
    """
    Return:
      thetas: [T]
      dtheta: [T-1] interval lengths for STR length accumulation
    """
    theta_min = float(theta_min)
    theta_max = float(theta_max)
    theta_points = int(theta_points)

    if theta_points < 2:
        raise ValueError("theta_points must be >= 2")

    if logspace:
        if theta_min <= 0.0:
            raise ValueError("theta_min must be > 0 for logspace grid.")
        thetas = np.logspace(np.log10(theta_min), np.log10(theta_max), theta_points)
    else:
        thetas = np.linspace(theta_min, theta_max, theta_points)

    dtheta = np.diff(thetas)  # nonuniform allowed
    return thetas.astype(np.float64), dtheta.astype(np.float64)


# ============================================================
# Cardinality sweep metrics
# ============================================================

def theta_sweep_metrics_hard(p: np.ndarray, k_true: np.ndarray, thetas: np.ndarray):
    """
    hard threshold:
      k_hat(theta) = #{i : p_i >= theta}
    """
    k_true = k_true.astype(np.int64)
    KMAE, KAcc = [], []
    for th in thetas:
        k_hat = (p >= th).sum(axis=1).astype(np.int64)
        KMAE.append(np.mean(np.abs(k_hat - k_true)))
        KAcc.append(np.mean((k_hat == k_true).astype(np.float32)))
    return np.array(KMAE, dtype=np.float64), np.array(KAcc, dtype=np.float64)


def theta_sweep_metrics_soft(p: np.ndarray, k_true: np.ndarray, thetas: np.ndarray, temp: float, eps: float = 1e-12):
    """
    soft-card surrogate aligned with training:
      k_hat_soft(theta) = sum_i sigmoid((p_i - theta)/T)
    """
    T = max(float(temp), eps)
    k_true_f = k_true.astype(np.float32)

    KMAE, KMSE = [], []
    for th in thetas:
        logits = (p - float(th)) / T
        s = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        k_hat = s.sum(axis=1).astype(np.float32)
        KMAE.append(np.mean(np.abs(k_hat - k_true_f)))
        KMSE.append(np.mean((k_hat - k_true_f) ** 2))
    return np.array(KMAE, dtype=np.float64), np.array(KMSE, dtype=np.float64)


def stable_threshold_region_STR(p: np.ndarray, k_true: np.ndarray, thetas: np.ndarray, dtheta: np.ndarray, tol: float = 0.95) -> float:
    """
    STR definition with correct length on nonuniform theta grid.

    For each k:
      f_k(theta) = P(k_hat(theta)=k | k_true=k)
      f_k,max = max_theta f_k(theta)
      good(theta) = [ f_k(theta) >= tol * f_k,max ]
      STR(k) = length({theta : good})  (approximated by sum of interval lengths)
    Overall:
      STR = average over k in unique_k of  (k * STR(k)) / |unique_k|
    """
    k_true = k_true.astype(np.int64)
    unique_k = np.unique(k_true)
    unique_k = unique_k[unique_k > 0]

    if len(unique_k) == 0:
        return 0.0

    total = 0.0
    denom = float(len(unique_k))

    T = len(thetas)
    if dtheta.shape[0] != T - 1:
        raise ValueError("dtheta must be length (len(thetas)-1).")

    for k in unique_k:
        idx = np.where(k_true == k)[0]
        if len(idx) == 0:
            continue
        pk = p[idx]

        f = []
        for th in thetas:
            k_hat = (pk >= th).sum(axis=1).astype(np.int64)
            f.append(np.mean((k_hat == k).astype(np.float32)))
        f = np.array(f, dtype=np.float64)

        fmax = float(f.max())
        if fmax <= 1e-12:
            str_k = 0.0
        else:
            good = (f >= (tol * fmax)).astype(np.float64)  # [T]
            good_interval = good[:-1] * good[1:]  # [T-1]
            length = float((good_interval * dtheta).sum())
            str_k = length

        total += (float(k) * str_k)

    return total / denom


# ============================================================
# Multilabel F1 sweep metrics (NEW)
# ============================================================

def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return a / (b + eps)


def multilabel_metrics_at_theta(
    p: np.ndarray,
    y: np.ndarray,
    theta: float,
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Compute multilabel metrics at a given threshold theta.

    - micro precision/recall/F1 (global TP/FP/FN)
    - macro F1 (per-label F1 average; labels with no positives in both true+pred -> treated as 0)
    - example-based F1 (per-sample F1 average)
    """
    pred = (p >= float(theta))
    true = (y > 0.5)

    tp = np.logical_and(pred, true).sum().astype(np.float64)
    fp = np.logical_and(pred, np.logical_not(true)).sum().astype(np.float64)
    fn = np.logical_and(np.logical_not(pred), true).sum().astype(np.float64)

    micro_prec = float(tp / (tp + fp + eps))
    micro_rec  = float(tp / (tp + fn + eps))
    micro_f1   = float((2.0 * micro_prec * micro_rec) / (micro_prec + micro_rec + eps))

    # Macro-F1
    tp_l = np.logical_and(pred, true).sum(axis=0).astype(np.float64)
    fp_l = np.logical_and(pred, np.logical_not(true)).sum(axis=0).astype(np.float64)
    fn_l = np.logical_and(np.logical_not(pred), true).sum(axis=0).astype(np.float64)

    prec_l = _safe_div(tp_l, tp_l + fp_l, eps)
    rec_l  = _safe_div(tp_l, tp_l + fn_l, eps)
    f1_l   = _safe_div(2.0 * prec_l * rec_l, prec_l + rec_l, eps)
    macro_f1 = float(np.mean(f1_l))

    # Example-based F1
    inter = np.logical_and(pred, true).sum(axis=1).astype(np.float64)
    pred_sum = pred.sum(axis=1).astype(np.float64)
    true_sum = true.sum(axis=1).astype(np.float64)
    ex_f1 = _safe_div(2.0 * inter, pred_sum + true_sum, eps)
    example_f1 = float(np.mean(ex_f1))

    return {
        "micro_prec": micro_prec,
        "micro_rec": micro_rec,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "example_f1": example_f1,
    }


def theta_sweep_multilabel_f1(p: np.ndarray, y: np.ndarray, thetas: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Sweep over thetas and return arrays:
      micro_prec, micro_rec, micro_f1, macro_f1, example_f1
    """
    micro_prec = np.zeros(len(thetas), dtype=np.float64)
    micro_rec  = np.zeros(len(thetas), dtype=np.float64)
    micro_f1   = np.zeros(len(thetas), dtype=np.float64)
    macro_f1   = np.zeros(len(thetas), dtype=np.float64)
    example_f1 = np.zeros(len(thetas), dtype=np.float64)

    for i, th in enumerate(thetas):
        m = multilabel_metrics_at_theta(p, y, th)
        micro_prec[i] = m["micro_prec"]
        micro_rec[i]  = m["micro_rec"]
        micro_f1[i]   = m["micro_f1"]
        macro_f1[i]   = m["macro_f1"]
        example_f1[i] = m["example_f1"]

    return {
        "micro_prec": micro_prec,
        "micro_rec": micro_rec,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "example_f1": example_f1,
    }


def stable_region_length_from_curve(curve: np.ndarray, dtheta: np.ndarray, tol_frac: float = 0.98) -> float:
    """
    Generic STR-style length for a 1D curve metric(theta).
    - Let vmax = max(curve).
    - Good points: curve >= tol_frac * vmax.
    - Length is sum of dtheta where BOTH endpoints are good (nonuniform grid).
    """
    if len(curve) < 2:
        return 0.0
    vmax = float(np.max(curve))
    if vmax <= 1e-12:
        return 0.0
    good = (curve >= (tol_frac * vmax)).astype(np.float64)
    good_interval = good[:-1] * good[1:]
    if dtheta.shape[0] != len(curve) - 1:
        raise ValueError("dtheta must be length len(curve)-1")
    return float((good_interval * dtheta).sum())


# ============================================================
# Debug
# ============================================================

def _debug_khat_stats(p: np.ndarray, k_true: np.ndarray, theta: float, topn: int = 10):
    k_hat = (p >= theta).sum(axis=1).astype(np.int64)
    print(f"[Debug] k_true stats: mean={k_true.mean():.3f}, std={k_true.std():.3f}, min={k_true.min()}, max={k_true.max()}")
    print(f"[Debug] k_hat stats @ theta: theta={theta:.8f}, mean={k_hat.mean():.3f}, std={k_hat.std():.3f}, min={k_hat.min()}, max={k_hat.max()}")
    cnt = Counter(k_hat.tolist()).most_common(topn)
    print(f"[Debug] k_hat top counts (value,count): {cnt}")


# ---------- OPTIONAL: legacy gamma-cardinality metrics (older surrogate) ----------

def khat_gamma(p: np.ndarray, gamma: float, eps: float = 1e-12) -> np.ndarray:
    """
    legacy:
      k_hat^gamma(p) = sum_i (1 - exp(-p_i/gamma))
    """
    g = max(float(gamma), eps)
    return (1.0 - np.exp(-p / g)).sum(axis=1)


def card_metrics_gamma(p: np.ndarray, k_true: np.ndarray, gamma: float):
    k_true_f = k_true.astype(np.float32)
    k_hat = khat_gamma(p, gamma=gamma)
    kmae = np.mean(np.abs(k_hat - k_true_f))
    kmse = np.mean((k_hat - k_true_f) ** 2)
    return float(kmae), float(kmse), k_hat


# ============================================================
# Theta selection / loading (NEW)
# ============================================================

def load_theta_star(theta_star_from: str, key: str) -> float:
    data = np.load(theta_star_from)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {theta_star_from}. Available keys: {list(data.keys())[:30]} ...")
    return float(data[key])


def pick_theta_star(
    thetas: np.ndarray,
    curves: Dict[str, np.ndarray],
    KAcc: np.ndarray,
    mode: str
) -> Tuple[float, int]:
    """
    mode in {"kacc", "micro_f1", "macro_f1", "example_f1"}
    """
    mode = mode.lower()
    if mode == "kacc":
        idx = int(np.argmax(KAcc))
    elif mode == "micro_f1":
        idx = int(np.argmax(curves["micro_f1"]))
    elif mode == "macro_f1":
        idx = int(np.argmax(curves["macro_f1"]))
    elif mode == "example_f1":
        idx = int(np.argmax(curves["example_f1"]))
    else:
        raise ValueError(f"Unknown theta_star_mode: {mode}")
    return float(thetas[idx]), idx


# ============================================================
# Eval
# ============================================================

@torch.no_grad()
def run_eval(
    npz_path: str,
    ckpt_path: str,
    head: str,
    split: str,
    batch: int,
    fixed_beta: float,
    verbose_khat: bool,
    theta_min: float,
    theta_max: float,
    theta_points: int,
    theta_logspace: bool,
    soft_card_temp: float,
    gamma,
    # theta selection / usage
    theta_star_mode: str,
    theta_star_from: Optional[str],
    theta_star_key: str,
    f1_str_tol: float,
    out_npz: str,
):
    device = get_device()
    print(f"[Device] {device}")
    print(f"[Eval] data={npz_path}, ckpt={ckpt_path}, head={head}, split={split}")

    data = np.load(npz_path)
    d = int(data["d"])
    m = int(data["m"])

    ds = SyntheticDataset(npz_path, split)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    hidden = int(args.get("hidden", 1024))

    model_g = MLPLogits(d=d, m=m, hidden=hidden).to(device)
    model_g.load_state_dict(ckpt["model_g"])
    model_g.eval()

    beta_head = None
    if head == "beta_entmax" and ckpt.get("beta_head") is not None:
        beta_head = BetaHead(
            d=d,
            hidden=256,
            beta_min=0.0,
            beta_max=float(args.get("beta_max", 10.0)),
        ).to(device)
        beta_head.load_state_dict(ckpt["beta_head"])
        beta_head.eval()

    all_p, all_y, all_k = [], [], []

    for x, y, k in tqdm(loader, desc="Inference"):
        x = x.to(device)
        z = model_g(x)

        if head == "softmax":
            p = softmax_head(z)
        elif head == "sparsemax":
            p = sparsemax(z)
        elif head == "entmax15":
            p = entmax15(z)
        elif head == "beta_entmax":
            if beta_head is None:
                beta = torch.full((x.size(0), 1), float(fixed_beta), device=device, dtype=z.dtype)
            else:
                beta = beta_head(x)
            p = beta_entmax(z, beta, alpha=1.5, dim=-1, n_iter=30)
        else:
            raise ValueError(f"Unknown head: {head}")

        p = safe_prob_eval(p)

        all_p.append(p.detach().cpu().numpy())
        all_y.append(y.numpy())
        all_k.append(k.numpy())

    p = np.concatenate(all_p, axis=0).astype(np.float64)     # [N,m]
    y = np.concatenate(all_y, axis=0).astype(np.float64)     # [N,m]
    k_true = np.concatenate(all_k, axis=0).astype(np.int64)  # [N]

    # --------------------------------------------------------
    # Coverage + uniformity (oracle on true support)
    # --------------------------------------------------------
    p_t = torch.from_numpy(p)
    y_t = torch.from_numpy(y)

    mean_s_mass, cov_nll = compute_support_mass_stats(p_t, y_t, eps=1e-12)

    Ugap_oracle = compute_U_gap_oracle(p_t, y_t)
    UKL_oracle = compute_U_KL_oracle(p_t, y_t, eps=1e-12)

    Ugap_w = compute_U_gap_weighted(p_t, y_t)
    UKL_w = compute_U_KL_weighted(p_t, y_t, eps=1e-12)

    # --------------------------------------------------------
    # theta grid + sweep (cardinality + STR)
    # --------------------------------------------------------
    thetas, dtheta = make_theta_grid(theta_min, theta_max, theta_points, logspace=theta_logspace)

    KMAE, KAcc = theta_sweep_metrics_hard(p, k_true, thetas)
    STR = stable_threshold_region_STR(p, k_true, thetas, dtheta, tol=0.95)

    best_kmae_idx = int(KMAE.argmin())
    best_kacc_idx = int(KAcc.argmax())

    # soft-card metrics aligned with training
    SoftKMAE, SoftKMSE = theta_sweep_metrics_soft(p, k_true, thetas, temp=soft_card_temp)
    best_soft_kmae_idx = int(SoftKMAE.argmin())
    best_soft_kmse_idx = int(SoftKMSE.argmin())

    # --------------------------------------------------------
    # NEW: multilabel F1 sweep + F1-STR
    # --------------------------------------------------------
    f1_curves = theta_sweep_multilabel_f1(p, y, thetas)

    micro_prec_curve = f1_curves["micro_prec"]
    micro_rec_curve  = f1_curves["micro_rec"]
    micro_f1_curve   = f1_curves["micro_f1"]
    macro_f1_curve   = f1_curves["macro_f1"]
    example_f1_curve = f1_curves["example_f1"]

    micro_f1_best_idx = int(np.argmax(micro_f1_curve))
    macro_f1_best_idx = int(np.argmax(macro_f1_curve))
    example_f1_best_idx = int(np.argmax(example_f1_curve))

    theta_star_micro_f1 = float(thetas[micro_f1_best_idx])
    theta_star_macro_f1 = float(thetas[macro_f1_best_idx])
    theta_star_example_f1 = float(thetas[example_f1_best_idx])
    theta_star_kacc = float(thetas[best_kacc_idx])

    F1_STR_micro = stable_region_length_from_curve(micro_f1_curve, dtheta, tol_frac=float(f1_str_tol))
    F1_STR_macro = stable_region_length_from_curve(macro_f1_curve, dtheta, tol_frac=float(f1_str_tol))
    F1_STR_example = stable_region_length_from_curve(example_f1_curve, dtheta, tol_frac=float(f1_str_tol))

    # --------------------------------------------------------
    # Decide theta* for "pred-support uniformity" + single-point reporting
    #   - either load theta* from val NPZ (recommended for test),
    #   - or pick on current split by theta_star_mode.
    # --------------------------------------------------------
    if theta_star_from is not None:
        theta_star = load_theta_star(theta_star_from, theta_star_key)
        # find nearest index for printing curves at that theta
        idx_near = int(np.argmin(np.abs(thetas - theta_star)))
        theta_star = float(thetas[idx_near])  # snap to grid for consistency
        theta_star_idx = idx_near
        theta_star_source = f"loaded:{theta_star_from}:{theta_star_key}"
    else:
        theta_star, theta_star_idx = pick_theta_star(thetas, f1_curves, KAcc, mode=theta_star_mode)
        theta_star_source = f"picked_on_{split}:{theta_star_mode}"

    # single-point multilabel metrics at theta*
    m_theta = multilabel_metrics_at_theta(p, y, theta_star)

    # Pred-support uniformity at theta*
    mean_khat, mean_pred_mass, pred_UKL, pred_Ugap = pred_support_uniformity(p, theta=theta_star, eps=1e-12)

    # legacy gamma-card metrics (optional)
    KMAE_g = None
    KMSE_g = None

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------
    print("\n==== Metrics (Synthetic {}) ====".format(split))

    print("\n-- Coverage on TRUE support (must be reported) --")
    print(f"True-support mass  E[s_mass] (higher better): {mean_s_mass:.6f}")
    print(f"SupportNLL  E[-log(s_mass)] (lower better): {cov_nll:.6f}")

    print("\n-- Uniformity on TRUE support (oracle) --")
    print(f"U_gap_oracle (lower better): {Ugap_oracle:.6f}")
    print(f"U_KL_oracle  (lower better): {UKL_oracle:.6f}")

    print("\n-- Mass-weighted uniformity on TRUE support (oracle, coverage-aware) --")
    print(f"U_gap_weighted (lower better): {Ugap_w:.6f}")
    print(f"U_KL_weighted  (lower better): {UKL_w:.6f}")

    print("\n-- Cardinality / threshold robustness --")
    print(f"Best (hard) KMAE over theta: {KMAE.min():.6f} at theta={thetas[best_kmae_idx]:.8f}")
    print(f"Best (hard) KAcc over theta: {KAcc.max():.6f} at theta={thetas[best_kacc_idx]:.8f}")
    print(f"STR (higher better): {STR:.6f}")
    print(f"Best (soft) KMAE over theta: {SoftKMAE.min():.6f} at theta={thetas[best_soft_kmae_idx]:.8f}  (T={soft_card_temp:g})")
    print(f"Best (soft) KMSE over theta: {SoftKMSE.min():.6f} at theta={thetas[best_soft_kmse_idx]:.8f}  (T={soft_card_temp:g})")

    print("\n-- Multilabel F1 sweep (NEW) --")
    print(f"Best micro-F1 over theta: {micro_f1_curve[micro_f1_best_idx]:.6f} at theta={theta_star_micro_f1:.8f}")
    print(f"Best macro-F1 over theta: {macro_f1_curve[macro_f1_best_idx]:.6f} at theta={theta_star_macro_f1:.8f}")
    print(f"Best example-F1 over theta: {example_f1_curve[example_f1_best_idx]:.6f} at theta={theta_star_example_f1:.8f}")
    print(f"F1-STR_micro (tol={f1_str_tol:g}) (higher better): {F1_STR_micro:.6f}")
    print(f"F1-STR_macro (tol={f1_str_tol:g}) (higher better): {F1_STR_macro:.6f}")
    print(f"F1-STR_example (tol={f1_str_tol:g}) (higher better): {F1_STR_example:.6f}")

    print("\n-- Single-point metrics at theta* (for reporting; theta* source matters) --")
    print(f"theta* = {theta_star:.8f}  (source: {theta_star_source})")
    print(f"micro-P/R/F1 @ theta*: {m_theta['micro_prec']:.6f} / {m_theta['micro_rec']:.6f} / {m_theta['micro_f1']:.6f}")
    print(f"macro-F1 @ theta*: {m_theta['macro_f1']:.6f}")
    print(f"example-F1 @ theta*: {m_theta['example_f1']:.6f}")

    print("\n-- Pred-support uniformity at theta* --")
    print(f"E[k_hat(theta*)] : {mean_khat:.6f}")
    print(f"E[pred_mass(theta*)] : {mean_pred_mass:.6f}")
    print(f"Pred U_KL(theta*) (lower better): {pred_UKL:.6f}")
    print(f"Pred U_gap(theta*) (lower better): {pred_Ugap:.6f}")

    if gamma is not None:
        KMAE_g, KMSE_g, _ = card_metrics_gamma(p, k_true, gamma=float(gamma))
        print("\n-- Legacy gamma-card (optional) --")
        print(f"Gamma-card (legacy) KMAE: {KMAE_g:.6f}  (gamma={float(gamma):.6g})")
        print(f"Gamma-card (legacy) KMSE: {KMSE_g:.6f}  (gamma={float(gamma):.6g})")

    if verbose_khat:
        _debug_khat_stats(p, k_true, theta_star, topn=12)

    # --------------------------------------------------------
    # Save NPZ
    # --------------------------------------------------------
    out = {
        # meta
        "split": np.array([split]),
        "thetas": thetas,
        "theta_logspace": (1 if theta_logspace else 0),
        "f1_str_tol": float(f1_str_tol),

        # hard-threshold card
        "KMAE": KMAE,
        "KAcc": KAcc,

        # STR (KAcc conditional by k_true)
        "STR": float(STR),

        # soft-card aligned with training
        "soft_card_temp": float(soft_card_temp),
        "SoftKMAE": SoftKMAE,
        "SoftKMSE": SoftKMSE,

        # TRUE-support coverage
        "true_support_mass_mean": float(mean_s_mass),
        "true_support_cov_nll": float(cov_nll),

        # TRUE-support uniformity (oracle)
        "U_gap_oracle": float(Ugap_oracle),
        "U_KL_oracle": float(UKL_oracle),

        # TRUE-support uniformity (coverage-aware)
        "U_gap_weighted": float(Ugap_w),
        "U_KL_weighted": float(UKL_w),

        # --- NEW: multilabel curves ---
        "micro_prec_curve": micro_prec_curve,
        "micro_rec_curve": micro_rec_curve,
        "micro_f1_curve": micro_f1_curve,
        "macro_f1_curve": macro_f1_curve,
        "example_f1_curve": example_f1_curve,

        # --- NEW: best theta for each criterion (use these keys from val) ---
        "theta_star_micro_f1": float(theta_star_micro_f1),
        "theta_star_macro_f1": float(theta_star_macro_f1),
        "theta_star_example_f1": float(theta_star_example_f1),
        "theta_star_kacc": float(theta_star_kacc),

        "micro_f1_best": float(micro_f1_curve[micro_f1_best_idx]),
        "macro_f1_best": float(macro_f1_curve[macro_f1_best_idx]),
        "example_f1_best": float(example_f1_curve[example_f1_best_idx]),

        # --- NEW: F1-STR style lengths (global, not conditional on k) ---
        "F1_STR_micro": float(F1_STR_micro),
        "F1_STR_macro": float(F1_STR_macro),
        "F1_STR_example": float(F1_STR_example),

        # --- Single-point at theta* used for pred-uniformity reporting ---
        "theta_star_used": float(theta_star),
        "theta_star_used_source": np.array([theta_star_source]),

        "micro_prec_theta_star": float(m_theta["micro_prec"]),
        "micro_rec_theta_star": float(m_theta["micro_rec"]),
        "micro_f1_theta_star": float(m_theta["micro_f1"]),
        "macro_f1_theta_star": float(m_theta["macro_f1"]),
        "example_f1_theta_star": float(m_theta["example_f1"]),

        # Pred-support uniformity at theta*
        "pred_khat_mean_theta_star": float(mean_khat),
        "pred_mass_mean_theta_star": float(mean_pred_mass),
        "pred_U_KL_theta_star": float(pred_UKL),
        "pred_U_gap_theta_star": float(pred_Ugap),

        # legacy gamma card
        "gamma": (-1.0 if gamma is None else float(gamma)),
        "KMAE_gamma": (-1.0 if KMAE_g is None else float(KMAE_g)),
        "KMSE_gamma": (-1.0 if KMSE_g is None else float(KMSE_g)),
    }
    np.savez_compressed(out_npz, **out)
    print(f"\nSaved: {out_npz}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--head", type=str, required=True, choices=["softmax", "sparsemax", "entmax15", "beta_entmax"])
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fixed_beta", type=float, default=1.0)
    ap.add_argument("--theta_min", type=float, default=1e-6)
    ap.add_argument("--theta_max", type=float, default=1e-2)
    ap.add_argument("--theta_points", type=int, default=2001)
    ap.add_argument("--theta_logspace", action="store_true", help="Use log-spaced theta grid (theta_min must be > 0).")
    ap.add_argument("--soft_card_temp", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--theta_star_mode", type=str, default="kacc",
                    choices=["kacc", "micro_f1", "macro_f1", "example_f1"],
                    help="If theta_star_from is not provided, pick theta* on current split by this criterion.")
    ap.add_argument("--theta_star_from", type=str, default=None,
                    help="(Recommended for test) NPZ path from VAL run that contains theta_star_* keys.")
    ap.add_argument("--theta_star_key", type=str, default="theta_star_micro_f1",
                    help="Key in theta_star_from NPZ to load. e.g., theta_star_micro_f1 / theta_star_macro_f1 / theta_star_kacc")
    ap.add_argument("--f1_str_tol", type=float, default=0.98,
                    help="F1-STR tolerance fraction: good if F1(theta) >= tol * max F1. (e.g., 0.98)")

    ap.add_argument("--verbose_khat", action="store_true")
    ap.add_argument("--out_npz", type=str, default="eval_results.npz")
    args = ap.parse_args()

    run_eval(
        npz_path=args.data,
        ckpt_path=args.ckpt,
        head=args.head,
        split=args.split,
        batch=args.batch,
        fixed_beta=args.fixed_beta,
        verbose_khat=args.verbose_khat,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_points=args.theta_points,
        theta_logspace=args.theta_logspace,
        soft_card_temp=args.soft_card_temp,
        gamma=args.gamma,
        theta_star_mode=args.theta_star_mode,
        theta_star_from=args.theta_star_from,
        theta_star_key=args.theta_star_key,
        f1_str_tol=args.f1_str_tol,
        out_npz=args.out_npz,
    )


if __name__ == "__main__":
    main()

"""
# --------------------
# 0) Run on VAL to select theta* (NO leakage)
# --------------------

# softmax (val)
python3 eval_metrics.py --data ./Synthetic.npz --ckpt ./ckpt_softmax.pt --head softmax \
  --split val \
  --theta_min 0.001 --theta_max 0.5 --theta_points 2001 \
  --soft_card_temp 1e-3 \
  --f1_str_tol 0.98 \
  --out_npz ./val_softmax_widetheta.npz

# sparsemax (val)
python3 eval_metrics.py --data ./Synthetic.npz --ckpt ./ckpt_sparsemax.pt --head sparsemax \
  --split val \
  --theta_min 0.001 --theta_max 0.5 --theta_points 2001 \
  --soft_card_temp 1e-3 \
  --f1_str_tol 0.98 \
  --out_npz ./val_sparsemax_widetheta.npz

# entmax15 (val)
python3 eval_metrics.py --data ./Synthetic.npz --ckpt ./ckpt_entmax15.pt --head entmax15 \
  --split val \
  --theta_min 0.001 --theta_max 0.5 --theta_points 2001 \
  --soft_card_temp 1e-3 \
  --f1_str_tol 0.98 \
  --out_npz ./val_entmax15_widetheta.npz

# beta-entmax ABL2 (no-card) (val)
python3 eval_metrics.py --data ./Synthetic.npz --ckpt ./ckpt_beta_abl2_nocard.pt --head beta_entmax \
  --split val \
  --theta_min 0.001 --theta_max 0.5 --theta_points 2001 \
  --soft_card_temp 1e-3 \
  --f1_str_tol 0.98 \
  --out_npz ./val_beta_abl2_nocard_widetheta.npz


# --------------------
# 1) Run on TEST using theta* loaded from VAL (NO leakage)
# --------------------

# softmax (test)
python3 eval_metrics.py --data ./Synthetic.npz --ckpt ./ckpt_softmax.pt --head softmax \
  --split test \
  --theta_min 0.001 --theta_max 0.5 --theta_points 2001 \
  --soft_card_temp 1e-3 \
  --theta_star_from ./val_softmax_widetheta.npz --theta_star_key theta_star_micro_f1 \
  --verbose_khat \
  --out_npz ./test_softmax_widetheta.npz

# sparsemax (test)
python3 eval_metrics.py --data ./Synthetic.npz --ckpt ./ckpt_sparsemax.pt --head sparsemax \
  --split test \
  --theta_min 0.001 --theta_max 0.5 --theta_points 2001 \
  --soft_card_temp 1e-3 \
  --theta_star_from ./val_sparsemax_widetheta.npz --theta_star_key theta_star_micro_f1 \
  --verbose_khat \
  --out_npz ./test_sparsemax_widetheta.npz

# entmax15 (test)
python3 eval_metrics.py --data ./Synthetic.npz --ckpt ./ckpt_entmax15.pt --head entmax15 \
  --split test \
  --theta_min 0.001 --theta_max 0.5 --theta_points 2001 \
  --soft_card_temp 1e-3 \
  --theta_star_from ./val_entmax15_widetheta.npz --theta_star_key theta_star_micro_f1 \
  --verbose_khat \
  --out_npz ./test_entmax15_widetheta.npz

# beta-entmax ABL2 (test)
python3 eval_metrics.py --data ./Synthetic.npz --ckpt ./ckpt_beta_abl2_nocard.pt --head beta_entmax \
  --split test \
  --theta_min 0.001 --theta_max 0.5 --theta_points 2001 \
  --soft_card_temp 1e-3 \
  --theta_star_from ./val_beta_abl2_nocard_widetheta.npz --theta_star_key theta_star_micro_f1 \
  --verbose_khat \
  --out_npz ./test_beta_abl2_nocard_widetheta.npz

"""
