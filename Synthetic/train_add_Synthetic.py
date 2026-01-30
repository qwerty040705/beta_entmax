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
from typing import Optional


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


def safe_sigmoid(p: torch.Tensor) -> torch.Tensor:
    p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    return p.clamp(0.0, 1.0)


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
# Model
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


# ============================================================
# Losses: BCE / ASL
# ============================================================

class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x_sigmoid = torch.sigmoid(logits)
        x_sigmoid = torch.nan_to_num(x_sigmoid, nan=0.0, posinf=1.0, neginf=0.0)

        # probs for positive/negative
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric clipping (for negatives)
        if self.clip is not None and self.clip > 0:
            xs_neg = torch.clamp(xs_neg + self.clip, max=1.0)

        # Basic CE
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg  # negative values

        # Focal weights
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            pt_pos = xs_pos * targets
            pt_neg = xs_neg * (1.0 - targets)
            pt = pt_pos + pt_neg

            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            w = torch.pow(1.0 - pt, gamma)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)

            loss = loss * w

        return -loss.mean()


def bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y)


# ============================================================
# Metrics (val reporting, comparable to train_new_Synthetic.py)
# ============================================================

@torch.no_grad()
def compute_metrics_from_loader_sigmoid(
    model_g,
    loader,
    theta_min: float,
    theta_max: float,
    theta_points: int,
    device,
):
    model_g.eval()

    all_p = []
    all_y = []
    all_k = []

    for x, y, k in loader:
        x = x.to(device)
        y = y.to(device)
        k = k.to(device)

        z = model_g(x)
        p = torch.sigmoid(z)
        p = safe_sigmoid(p)

        all_p.append(p.detach().cpu())
        all_y.append(y.detach().cpu())
        all_k.append(k.detach().cpu())

    if not all_p:
        return {}

    P = torch.cat(all_p, dim=0)  # [N,m]
    Y = torch.cat(all_y, dim=0)  # [N,m]
    K = torch.cat(all_k, dim=0)  # [N]

    mask = (Y > 0.5)


    pos_min = P.masked_fill(~mask, float("inf")).min(dim=-1).values
    neg_max = P.masked_fill(mask, float("-inf")).max(dim=-1).values
    neg_max = torch.where(torch.isfinite(neg_max), neg_max, torch.zeros_like(neg_max))
    pos_min = torch.where(torch.isfinite(pos_min), pos_min, torch.zeros_like(pos_min))
    STR = (pos_min - neg_max).clamp(min=0.0).mean().item()

    kf = mask.sum(dim=-1).clamp(min=1).to(P.dtype)
    avg_pos_prob = (P * mask.to(P.dtype)).sum(dim=-1) / kf
    LEAK = (1.0 - avg_pos_prob).clamp(min=0.0).mean().item()

    pS = P * mask.to(P.dtype)
    s_mass = pS.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    p_tilde = pS / s_mass
    target = (1.0 / kf).unsqueeze(-1)
    U_L2 = ((((p_tilde - target) ** 2) * mask.to(P.dtype)).sum(dim=-1) / kf).mean().item()

    tmin = max(theta_min, 1e-8)
    tmax = max(theta_max, tmin * 10.0)

    if tmax <= 1.0 and tmin >= 1e-8 and (tmax / tmin) > 10.0:
        thetas = torch.logspace(np.log10(tmin), np.log10(tmax), steps=int(theta_points), device="cpu", dtype=P.dtype)
    else:
        thetas = torch.linspace(tmin, min(tmax, 1.0), steps=int(theta_points), device="cpu", dtype=P.dtype)

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
# Train / Eval
# ============================================================

@torch.no_grad()
def evaluate_loss(model_g, loader, loss_name: str, asl_obj: Optional[AsymmetricLoss], device):
    model_g.eval()
    losses = []
    for x, y, _k in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model_g(x)

        if loss_name == "bce":
            loss = bce_loss(logits, y)
        elif loss_name == "asl":
            loss = asl_obj(logits, y)
        else:
            raise ValueError(loss_name)

        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def train_one_epoch(model_g, opt, loader, loss_name: str, asl_obj: Optional[AsymmetricLoss], device, grad_clip: float):
    model_g.train()
    losses = []
    t0 = time.time()
    pbar = tqdm(loader, desc="Train", leave=False)

    for it, (x, y, _k) in enumerate(pbar, start=1):
        x = x.to(device)
        y = y.to(device)

        logits = model_g(x)

        if loss_name == "bce":
            loss = bce_loss(logits, y)
        elif loss_name == "asl":
            loss = asl_obj(logits, y)
        else:
            raise ValueError(loss_name)

        if not torch.isfinite(loss):
            print("[WARN] Non-finite loss detected. Skipping batch.")
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model_g.parameters(), grad_clip)

        opt.step()

        losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))

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
    loss: str
    epochs: int
    batch: int
    lr: float
    wd: float
    hidden: int
    seed: int

    gamma_pos: float
    gamma_neg: float
    clip: float

    grad_clip: float
    num_workers: int

    val_theta_min: float
    val_theta_max: float
    val_theta_points: int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="ckpt_add.pt")
    ap.add_argument("--loss", type=str, required=True, choices=["bce", "asl"])

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--gamma_pos", type=float, default=0.0)
    ap.add_argument("--gamma_neg", type=float, default=4.0)
    ap.add_argument("--clip", type=float, default=0.05)

    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--val_theta_min", type=float, default=0.05)
    ap.add_argument("--val_theta_max", type=float, default=0.95)
    ap.add_argument("--val_theta_points", type=int, default=31)

    args = ap.parse_args()

    targs = TrainArgs(
        data=args.data,
        out=args.out,
        loss=args.loss,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        wd=args.wd,
        hidden=args.hidden,
        seed=args.seed,
        gamma_pos=args.gamma_pos,
        gamma_neg=args.gamma_neg,
        clip=args.clip,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        val_theta_min=args.val_theta_min,
        val_theta_max=args.val_theta_max,
        val_theta_points=args.val_theta_points,
    )

    set_seed(targs.seed)
    device = get_device()
    print(f"[Device] {device}")
    print(f"[Train-Add] data={targs.data}, loss={targs.loss}, out={targs.out}")

    data_npz = np.load(targs.data)
    d = int(data_npz["d"])
    m = int(data_npz["m"])

    ds_train = SyntheticDataset(targs.data, "train")
    ds_val = SyntheticDataset(targs.data, "val")
    train_loader = DataLoader(ds_train, batch_size=targs.batch, shuffle=True, num_workers=targs.num_workers)
    val_loader = DataLoader(ds_val, batch_size=targs.batch, shuffle=False, num_workers=targs.num_workers)

    model_g = MLPLogits(d=d, m=m, hidden=targs.hidden).to(device)

    asl_obj = None
    if targs.loss == "asl":
        asl_obj = AsymmetricLoss(
            gamma_pos=float(targs.gamma_pos),
            gamma_neg=float(targs.gamma_neg),
            clip=float(targs.clip),
        ).to(device)
        print(f"[ASL] gamma_pos={targs.gamma_pos}, gamma_neg={targs.gamma_neg}, clip={targs.clip}")

    opt = torch.optim.AdamW(model_g.parameters(), lr=targs.lr, weight_decay=targs.wd)

    def score_from_metrics(mt: dict):
        if not mt:
            return -1e9
        return (1.0 * mt["F1_best"] + 0.5 * mt["STR"] + 0.5 * mt["KAcc_best"] - 0.1 * mt["Leak"])

    best_score = -1e9
    best_state = None
    epoch_times = []

    for epoch in range(1, targs.epochs + 1):
        ep_t0 = time.time()

        tr_loss = train_one_epoch(
            model_g=model_g,
            opt=opt,
            loader=train_loader,
            loss_name=targs.loss,
            asl_obj=asl_obj,
            device=device,
            grad_clip=targs.grad_clip,
        )

        va_loss = evaluate_loss(
            model_g=model_g,
            loader=val_loader,
            loss_name=targs.loss,
            asl_obj=asl_obj,
            device=device,
        )

        metrics = compute_metrics_from_loader_sigmoid(
            model_g=model_g,
            loader=val_loader,
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
            f"theta*={metrics.get('theta_best',0):.3f} | ETA_total={eta_total:,.1f}s"
        )
        print(msg)

        sc = score_from_metrics(metrics)
        if sc > best_score:
            best_score = sc
            best_state = {
                "model_g": model_g.state_dict(),
                "args": asdict(targs),
                "best_score": best_score,
                "best_metrics": metrics,
                "loss_kind": targs.loss,
                "asl_params": None if asl_obj is None else {
                    "gamma_pos": targs.gamma_pos,
                    "gamma_neg": targs.gamma_neg,
                    "clip": targs.clip,
                },
            }

    if best_state is None:
        best_state = {
            "model_g": model_g.state_dict(),
            "args": asdict(targs),
            "best_score": best_score,
            "best_metrics": {},
            "loss_kind": targs.loss,
            "asl_params": None,
        }

    os.makedirs(os.path.dirname(targs.out) or ".", exist_ok=True)
    torch.save(best_state, targs.out)
    print(f"[Saved] {targs.out}  (best_score={best_score:.6f})")
    print("[Best metrics]", best_state.get("best_metrics", {}))


if __name__ == "__main__":
    main()


"""
# ============================================================
# Example commands
# ============================================================

# BCE sigmoid baseline
python3 train_add_Synthetic.py --data ./Synthetic.npz --out ckpt_bce.pt --loss bce --epochs 60 \
  --lr 1e-4 --batch 256 --hidden 1024 \
  --val_theta_min 0.05 --val_theta_max 0.95 --val_theta_points 31

# ASL baseline (common setting: gamma_neg=4, gamma_pos=0, clip=0.05)
python3 train_add_Synthetic.py --data ./Synthetic.npz --out ckpt_asl.pt --loss asl --epochs 60 \
  --gamma_pos 0 --gamma_neg 4 --clip 0.05 \
  --lr 1e-4 --batch 256 --hidden 1024 \
  --val_theta_min 0.05 --val_theta_max 0.95 --val_theta_points 31
"""
