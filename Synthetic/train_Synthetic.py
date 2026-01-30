import argparse
import os
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
# Heads: sparsemax / entmax1.5 / beta-entmax (same as yours)
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
# Losses
# ============================================================

def multilabel_distribution(y: torch.Tensor) -> torch.Tensor:
    k = y.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return y / k


def loss_task_ce(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    q = multilabel_distribution(y)
    p = safe_prob(p)
    return -(q * torch.log(p.clamp(min=eps))).sum(dim=-1).mean()


def loss_uniform_kl(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = safe_prob(p)
    mask = y > 0.5
    k = mask.sum(dim=1).clamp(min=1).to(p.dtype)  # [B]

    pS = p * mask.to(p.dtype)
    s_mass = pS.sum(dim=1, keepdim=True).clamp(min=eps)
    p_tilde = (pS / s_mass).clamp(min=eps)

    log_p = torch.log(p_tilde)
    log_u = -torch.log(k).unsqueeze(1)  # log(1/k)

    kl = (p_tilde * (log_p - log_u) * mask.to(p.dtype)).sum(dim=1)
    return kl.mean()


def loss_separation_margin(p: torch.Tensor, y: torch.Tensor, margin: float = 0.01) -> torch.Tensor:
    p = safe_prob(p)
    mask = y > 0.5

    # min positive prob
    pos_min = p.masked_fill(~mask, float("inf")).min(dim=1).values
    # max negative prob
    neg_max = p.masked_fill(mask, float("-inf")).max(dim=1).values

    # hinge
    return F.relu(margin + neg_max - pos_min).mean()


def soft_count_theta(p: torch.Tensor, theta: float, temp: float) -> torch.Tensor:
    return torch.sigmoid((p - float(theta)) / float(temp)).sum(dim=-1)


def loss_card_multi_theta(p: torch.Tensor, k_true: torch.Tensor,
                          theta_min: float, theta_max: float, theta_points: int,
                          temp: float = 1e-3) -> torch.Tensor:
    p = safe_prob(p)
    k_true_f = k_true.to(p.dtype)

    tmin = max(theta_min, 1e-6)
    tmax = max(theta_max, tmin * 10.0)

    thetas = torch.logspace(
        np.log10(tmin), 
        np.log10(tmax), 
        steps=int(theta_points),
        dtype=p.dtype, 
        device="cpu"  
    ).to(p.device)   

    losses = []
    for th in thetas:
        k_hat = soft_count_theta(p, theta=float(th.item()), temp=temp)
        losses.append(F.mse_loss(k_hat, k_true_f))
    
    return torch.stack(losses).mean()


def beta_reg_loss(beta: torch.Tensor) -> torch.Tensor:
    return (beta ** 2).mean()


# ============================================================
# Train / Eval loops
# ============================================================

@torch.no_grad()
def evaluate(model_g, beta_head, loader,
             head: str, fixed_beta: float,
             # losses
             lambda_uni: float,
             lambda_sep: float, sep_margin: float,
             card_loss_on: int, lambda_card: float,
             card_theta_min: float, card_theta_max: float, card_theta_points: int, card_temp: float,
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

        p = safe_prob(p)

        loss = loss_task_ce(p, y)

        if lambda_uni > 0:
            loss = loss + lambda_uni * loss_uniform_kl(p, y)

        if lambda_sep > 0:
            loss = loss + lambda_sep * loss_separation_margin(p, y, margin=sep_margin)

        if card_loss_on:
            loss = loss + lambda_card * loss_card_multi_theta(
                p, k,
                theta_min=card_theta_min, theta_max=card_theta_max,
                theta_points=card_theta_points,
                temp=card_temp
            )

        if head == "beta_entmax" and (beta_head is not None) and (eta_beta > 0.0):
            loss = loss + eta_beta * beta_reg_loss(beta)

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else 0.0


def train_one_epoch(model_g, beta_head, opt, loader,
                    head: str, fixed_beta: float,
                    lambda_uni: float,
                    lambda_sep: float, sep_margin: float,
                    card_loss_on: int, lambda_card: float,
                    card_theta_min: float, card_theta_max: float, card_theta_points: int, card_temp: float,
                    eta_beta: float,
                    device, grad_clip: float):
    model_g.train()
    if beta_head is not None:
        beta_head.train()

    losses = []
    for x, y, k in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)
        k = k.to(device)

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

        p = safe_prob(p)

        loss = loss_task_ce(p, y)

        if lambda_uni > 0:
            loss = loss + lambda_uni * loss_uniform_kl(p, y)

        if lambda_sep > 0:
            loss = loss + lambda_sep * loss_separation_margin(p, y, margin=sep_margin)

        if card_loss_on:
            loss = loss + lambda_card * loss_card_multi_theta(
                p, k,
                theta_min=card_theta_min, theta_max=card_theta_max,
                theta_points=card_theta_points,
                temp=card_temp
            )

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

    lambda_uni: float

    lambda_sep: float
    sep_margin: float

    card_loss: int
    lambda_card: float
    card_theta_min: float
    card_theta_max: float
    card_theta_points: int
    card_temp: float

    eta_beta: float
    grad_clip: float
    num_workers: int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="ckpt.pt")
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

    ap.add_argument("--lambda_uni", type=float, default=0.1)       
    ap.add_argument("--lambda_sep", type=float, default=1.0)        
    ap.add_argument("--sep_margin", type=float, default=0.01)      

    ap.add_argument("--card_loss", type=int, default=1)           
    ap.add_argument("--lambda_card", type=float, default=1.0)
    ap.add_argument("--card_theta_min", type=float, default=1e-6)
    ap.add_argument("--card_theta_max", type=float, default=1e-2)
    ap.add_argument("--card_theta_points", type=int, default=12)    
    ap.add_argument("--card_temp", type=float, default=1e-3)       

    ap.add_argument("--eta_beta", type=float, default=1e-3)        
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=0)
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
        lambda_uni=args.lambda_uni,
        lambda_sep=args.lambda_sep,
        sep_margin=args.sep_margin,
        card_loss=args.card_loss,
        lambda_card=args.lambda_card,
        card_theta_min=args.card_theta_min,
        card_theta_max=args.card_theta_max,
        card_theta_points=args.card_theta_points,
        card_temp=args.card_temp,
        eta_beta=args.eta_beta,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
    )

    set_seed(targs.seed)
    device = get_device()
    print(f"[Device] {device}")
    print(f"[Train] data={targs.data}, head={targs.head}, out={targs.out}")
    print(f"[Train] lambda_uni={targs.lambda_uni}, lambda_sep={targs.lambda_sep}, sep_margin={targs.sep_margin}")
    print(f"[Train] card_loss={targs.card_loss}, lambda_card={targs.lambda_card}, "
          f"theta=[{targs.card_theta_min},{targs.card_theta_max}], points={targs.card_theta_points}, temp={targs.card_temp}")
    if targs.head == "beta_entmax":
        print(f"[Train] train_beta_head={targs.train_beta_head}, fixed_beta={targs.fixed_beta}, beta_max={targs.beta_max}, eta_beta={targs.eta_beta}")

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

    best_val = float("inf")
    best_state = None

    for epoch in range(1, targs.epochs + 1):
        tr_loss = train_one_epoch(
            model_g=model_g,
            beta_head=beta_head,
            opt=opt,
            loader=train_loader,
            head=targs.head,
            fixed_beta=targs.fixed_beta,
            lambda_uni=targs.lambda_uni,
            lambda_sep=targs.lambda_sep,
            sep_margin=targs.sep_margin,
            card_loss_on=targs.card_loss,
            lambda_card=targs.lambda_card,
            card_theta_min=targs.card_theta_min,
            card_theta_max=targs.card_theta_max,
            card_theta_points=targs.card_theta_points,
            card_temp=targs.card_temp,
            eta_beta=targs.eta_beta,
            device=device,
            grad_clip=targs.grad_clip,
        )

        va_loss = evaluate(
            model_g=model_g,
            beta_head=beta_head,
            loader=val_loader,
            head=targs.head,
            fixed_beta=targs.fixed_beta,
            lambda_uni=targs.lambda_uni,
            lambda_sep=targs.lambda_sep,
            sep_margin=targs.sep_margin,
            card_loss_on=targs.card_loss,
            lambda_card=targs.lambda_card,
            card_theta_min=targs.card_theta_min,
            card_theta_max=targs.card_theta_max,
            card_theta_points=targs.card_theta_points,
            card_temp=targs.card_temp,
            eta_beta=targs.eta_beta,
            device=device,
        )

        print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.6f}  val_loss={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {
                "model_g": model_g.state_dict(),
                "beta_head": None if beta_head is None else beta_head.state_dict(),
                "args": asdict(targs),
            }

    if best_state is None:
        best_state = {
            "model_g": model_g.state_dict(),
            "beta_head": None if beta_head is None else beta_head.state_dict(),
            "args": asdict(targs),
        }

    os.makedirs(os.path.dirname(targs.out) or ".", exist_ok=True)
    torch.save(best_state, targs.out)
    print(f"[Saved] {targs.out}  (best_val={best_val:.6f})")


if __name__ == "__main__":
    main()



"""
# --------------------
# Main comparison
# --------------------

# 1) softmax (baseline)
python3 train_Synthetic.py --data ./Synthetic.npz --out ckpt_softmax.pt --head softmax --epochs 20 \
  --card_loss 0 --lambda_uni 0.0 --lambda_sep 0.0

# 2) sparsemax (baseline)
python3 train_Synthetic.py --data ./Synthetic.npz --out ckpt_sparsemax.pt --head sparsemax --epochs 20 \
  --card_loss 0 --lambda_uni 0.0 --lambda_sep 0.0

# 3) entmax (entmax1.5 baseline)
python3 train_Synthetic.py --data ./Synthetic.npz --out ckpt_entmax15.pt --head entmax15 --epochs 20 \
  --card_loss 0 --lambda_uni 0.0 --lambda_sep 0.0

# 4) β-entmax (trained β + multi-theta cardinality ON + separation ON + uniformity ON)
python3 train_Synthetic.py --data ./Synthetic.npz --out ckpt_beta_main.pt --head beta_entmax --epochs 20 \
  --train_beta_head 1 --beta_max 10.0 \
  --lambda_uni 0.1 \
  --lambda_sep 1.0 --sep_margin 0.01 \
  --card_loss 1 --lambda_card 1.0 \
  --card_theta_min 1e-6 --card_theta_max 1e-2 --card_theta_points 12 --card_temp 1e-3 \
  --eta_beta 1e-3


# --------------------
# Ablation
# --------------------

# Ablation 1) fixed β + (card ON, sep ON, uni ON)
python3 train_Synthetic.py --data ./Synthetic.npz --out ckpt_beta_abl1_fixedbeta.pt --head beta_entmax --epochs 20 \
  --train_beta_head 0 --fixed_beta 1.0 \
  --lambda_uni 0.1 \
  --lambda_sep 1.0 --sep_margin 0.01 \
  --card_loss 1 --lambda_card 1.0 \
  --card_theta_min 1e-6 --card_theta_max 1e-2 --card_theta_points 12 --card_temp 1e-3 \
  --eta_beta 0.0


# Ablation 2) trained β + (card OFF) + (sep ON, uni ON) : cardinality alignment 제거
python3 train_Synthetic.py --data ./Synthetic.npz --out ckpt_beta_abl2_nocard.pt --head beta_entmax --epochs 20 \
  --train_beta_head 1 --beta_max 10.0 \
  --lambda_uni 0.1 \
  --lambda_sep 1.0 --sep_margin 0.01 \
  --card_loss 0 \
  --eta_beta 1e-3


# Ablation 3) trained β + (sep OFF) + (card ON, uni ON) : separation 제거
python3 train_Synthetic.py --data ./Synthetic.npz --out ckpt_beta_abl3_nosep.pt --head beta_entmax --epochs 20 \
  --train_beta_head 1 --beta_max 10.0 \
  --lambda_uni 0.1 \
  --lambda_sep 0.0 \
  --card_loss 1 --lambda_card 1.0 \
  --card_theta_min 1e-6 --card_theta_max 1e-2 --card_theta_points 12 --card_temp 1e-3 \
  --eta_beta 1e-3


# Ablation 4) trained β + (uni OFF) + (card ON, sep ON) : uniformity 제거
python3 train_Synthetic.py --data ./Synthetic.npz --out ckpt_beta_abl4_nouni.pt --head beta_entmax --epochs 20 \
  --train_beta_head 1 --beta_max 10.0 \
  --lambda_uni 0.0 \
  --lambda_sep 1.0 --sep_margin 0.01 \
  --card_loss 1 --lambda_card 1.0 \
  --card_theta_min 1e-6 --card_theta_max 1e-2 --card_theta_points 12 --card_temp 1e-3 \
  --eta_beta 1e-3


# Ablation 5) trained β + (card ON, sep ON, uni ON) but weaker sep 
python3 train_Synthetic.py --data ./Synthetic.npz --out ckpt_beta_abl5_margin005.pt --head beta_entmax --epochs 20 \
  --train_beta_head 1 --beta_max 10.0 \
  --lambda_uni 0.1 \
  --lambda_sep 1.0 --sep_margin 0.005 \
  --card_loss 1 --lambda_card 1.0 \
  --card_theta_min 1e-6 --card_theta_max 1e-2 --card_theta_points 12 --card_temp 1e-3 \
  --eta_beta 1e-3
"""
