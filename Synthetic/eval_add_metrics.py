import argparse
import os
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
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
    Convert nonnegative scores to a probability distribution (sum=1).
    - nan/inf -> 0, negatives -> 0
    - renorm to sum=1; if sum=0 -> uniform
    """
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = p.clamp(min=0.0)
    s = p.sum(dim=-1, keepdim=True)
    m = p.size(-1)
    return torch.where(s > eps_sum, p / (s + eps_sum), torch.full_like(p, 1.0 / m))


def np_logspace(minv: float, maxv: float, n: int, logspace: bool):
    minv = max(float(minv), 1e-12)
    maxv = max(float(maxv), minv * 1.000001)
    if logspace:
        return np.logspace(np.log10(minv), np.log10(maxv), int(n), dtype=np.float64)
    return np.linspace(minv, maxv, int(n), dtype=np.float64)


# ============================================================
# Dataset
# ============================================================

class NpzMultilabelDataset(Dataset):
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
# Model (same backbone as your training scripts)
# ============================================================

class MLPLogits(nn.Module):
    def __init__(self, d: int, m: int, hidden: int = 1024):
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


def load_ckpt_build_model(ckpt_path: str, d: int, m: int, hidden: int, device) -> nn.Module:
    """
    Robust loader:
    - supports torch.save(state_dict)
    - supports torch.save({"model_g": state_dict, ...})
    - supports torch.save({"model": state_dict, ...})
    """
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        if "model_g" in obj and isinstance(obj["model_g"], dict):
            state = obj["model_g"]
        elif "model" in obj and isinstance(obj["model"], dict):
            state = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        else:
            # maybe it's already a state dict
            state = obj
    else:
        state = obj

    model = MLPLogits(d=d, m=m, hidden=hidden).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing keys: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    model.eval()
    return model


# ============================================================
# Metrics
# ============================================================

def micro_prf1(pred: torch.Tensor, y: torch.Tensor):
    # pred,y: bool tensors [N,m]
    TP = (pred & y).sum().item()
    FP = (pred & ~y).sum().item()
    FN = ((~pred) & y).sum().item()
    P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    return P, R, F1


def macro_f1(pred: torch.Tensor, y: torch.Tensor, eps: float = 1e-12):
    pred_f = pred.float()
    y_f = y.float()
    TP = (pred_f * y_f).sum(dim=0)
    FP = (pred_f * (1.0 - y_f)).sum(dim=0)
    FN = ((1.0 - pred_f) * y_f).sum(dim=0)
    f1 = (2 * TP) / (2 * TP + FP + FN + eps)
    return f1.mean().item()


def example_f1(pred: torch.Tensor, y: torch.Tensor, eps: float = 1e-12):
    pred_f = pred.float()
    y_f = y.float()
    TP = (pred_f * y_f).sum(dim=1)
    FP = (pred_f * (1.0 - y_f)).sum(dim=1)
    FN = ((1.0 - pred_f) * y_f).sum(dim=1)
    f1 = (2 * TP) / (2 * TP + FP + FN + eps)
    return f1.mean().item()


def kl_from_pu(p: torch.Tensor, u: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (p.clamp(min=eps) * (torch.log(p.clamp(min=eps)) - torch.log(u.clamp(min=eps)))).sum(dim=-1)


@torch.no_grad()
def compute_oracle_uniformity_and_mass(p_dist: torch.Tensor, y_mask: torch.Tensor):
    """
    UNIFORMITY/MASS metrics are computed on p_dist (sum=1) for fair comparison.
    p_dist: [N,m] distribution (sum=1)
    y_mask: [N,m] bool mask of true labels
    Returns:
      s_mass_mean, support_nll, U_gap_oracle, U_KL_oracle,
      U_gap_weighted, U_KL_weighted
    """
    mask_f = y_mask.to(p_dist.dtype)
    k = mask_f.sum(dim=-1).clamp(min=1.0)

    pS = p_dist * mask_f
    s_mass = pS.sum(dim=-1).clamp(min=1e-12)  # [N]
    support_nll = (-torch.log(s_mass)).mean().item()

    # renorm on TRUE support S
    p_tilde = pS / s_mass.unsqueeze(-1)

    # uniform target on TRUE support
    u = (mask_f / k.unsqueeze(-1)).clamp(min=0.0)

    # L1 gap to uniform (on support)
    gap_vec = (torch.abs(p_tilde - u) * mask_f).sum(dim=-1)
    gap = gap_vec.mean().item()

    # KL(p_tilde || u) on support
    kl_vec = kl_from_pu(p_tilde, u)
    kl = kl_vec.mean().item()

    # coverage-aware weighting by s_mass
    w = s_mass.detach()
    w_sum = w.sum().clamp(min=1e-12)
    gap_w = (gap_vec * w).sum().item() / w_sum.item()
    kl_w = (kl_vec * w).sum().item() / w_sum.item()

    return {
        "s_mass_mean": s_mass.mean().item(),
        "support_nll": support_nll,
        "U_gap_oracle": gap,
        "U_KL_oracle": kl,
        "U_gap_weighted": gap_w,
        "U_KL_weighted": kl_w,
    }


@torch.no_grad()
def pred_support_uniformity(p_dist: torch.Tensor, pred: torch.Tensor):
    """
    Pred-support uniformity at theta*, computed on p_dist for fair comparison.
    p_dist: [N,m] distribution (sum=1)
    pred: bool [N,m]
    returns: mean k_hat, pred_mass, Pred_U_KL, Pred_U_gap
    """
    N, m = p_dist.shape
    pred_f = pred.to(p_dist.dtype)

    pP = p_dist * pred_f
    pred_mass = pP.sum(dim=-1).clamp(min=0.0)

    # renorm within predicted support; if pred_mass==0 -> uniform over all labels
    p_tilde = torch.where(
        pred_mass.unsqueeze(-1) > 1e-12,
        pP / pred_mass.unsqueeze(-1),
        torch.full_like(pP, 1.0 / m),
    )

    # uniform target on predicted support; if empty -> uniform over all
    denom = pred_f.sum(dim=-1, keepdim=True)
    u = torch.where(
        denom > 0,
        pred_f / denom.clamp(min=1.0),
        torch.full_like(pred_f, 1.0 / m),
    ).to(p_dist.dtype)

    # only count gap on predicted support (when denom>0)
    gap = (torch.abs(p_tilde - u) * (pred_f > 0).to(p_dist.dtype)).sum(dim=-1).mean().item()
    kl = kl_from_pu(p_tilde, u).mean().item()

    return {
        "khat_mean": pred.sum(dim=-1).float().mean().item(),
        "pred_mass_mean": pred_mass.mean().item(),
        "Pred_U_KL": kl,
        "Pred_U_gap": gap,
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def run_eval(model: nn.Module,
             loader: DataLoader,
             theta_min: float, theta_max: float, theta_points: int,
             theta_logspace: bool,
             str_tol: float,
             f1_str_tol: float,
             theta_star_mode: str,
             theta_star_from: Optional[str],
             theta_star_key: str,
             verbose_khat: bool,
             device):
    """
    Implements the fair comparison policy:
      - score for threshold sweep: s = p_sig = sigmoid(z)  (natural score for sigmoid baseline)
      - distribution for uniformity/mass: p_dist = safe_prob(p_sig)  (sum=1)

    Returns dict of summary metrics + arrays.
    """

    model.eval()
    all_score = []   # s = p_sig for thresholding
    all_pdist = []   # p_dist for distribution metrics
    all_y = []
    all_k = []

    pbar = tqdm(loader, desc="Inference", leave=False)
    for x, y, k in pbar:
        x = x.to(device)
        z = model(x)

        # NATURAL score for sigmoid baseline decision:
        score = torch.sigmoid(z)          # [B,m] in (0,1)  (sum!=1)

        # FAIR distribution for uniformity/mass metrics:
        p_dist = safe_prob(score)         # [B,m] sum=1

        all_score.append(score.detach().cpu())
        all_pdist.append(p_dist.detach().cpu())
        all_y.append(y.detach().cpu())
        all_k.append(k.detach().cpu())

    S = torch.cat(all_score, dim=0)      # [N,m]  decision score for sweep
    P_dist = torch.cat(all_pdist, dim=0) # [N,m]  distribution for uniformity/mass
    Y = torch.cat(all_y, dim=0)          # [N,m]
    K = torch.cat(all_k, dim=0)          # [N]
    mask = (Y > 0.5)

    # ---- Oracle metrics on TRUE support using p_dist (fair / comparable)
    oracle = compute_oracle_uniformity_and_mass(P_dist, mask)

    # ---- Sweeps over theta on NATURAL score (sigmoid probs)
    thetas = np_logspace(theta_min, theta_max, theta_points, theta_logspace)

    micro_f1 = np.zeros_like(thetas, dtype=np.float64)
    macro_f1_arr = np.zeros_like(thetas, dtype=np.float64)
    example_f1_arr = np.zeros_like(thetas, dtype=np.float64)

    kmae = np.zeros_like(thetas, dtype=np.float64)
    kacc = np.zeros_like(thetas, dtype=np.float64)

    for i, th in enumerate(thetas):
        pred = (S >= float(th))
        Pm, Rm, Fm = micro_prf1(pred, mask)
        micro_f1[i] = Fm
        macro_f1_arr[i] = macro_f1(pred, mask)
        example_f1_arr[i] = example_f1(pred, mask)

        khat = pred.sum(dim=-1).to(torch.long)
        kacc[i] = (khat == K).float().mean().item()
        kmae[i] = (khat.to(torch.float32) - K.to(torch.float32)).abs().mean().item()

    # Best points
    i_best_micro = int(np.argmax(micro_f1))
    i_best_macro = int(np.argmax(macro_f1_arr))
    i_best_example = int(np.argmax(example_f1_arr))

    best_micro = float(micro_f1[i_best_micro])
    best_macro = float(macro_f1_arr[i_best_macro])
    best_example = float(example_f1_arr[i_best_example])

    theta_star_micro = float(thetas[i_best_micro])
    theta_star_macro = float(thetas[i_best_macro])
    theta_star_example = float(thetas[i_best_example])

    # Cardinality bests (hard)
    i_best_kmae = int(np.argmin(kmae))
    i_best_kacc = int(np.argmax(kacc))
    best_kmae = float(kmae[i_best_kmae]); theta_kmae = float(thetas[i_best_kmae])
    best_kacc = float(kacc[i_best_kacc]); theta_kacc = float(thetas[i_best_kacc])

    # STR (cardinality robustness): relative stable region around best KAcc
    STR = float(np.mean(kacc >= (float(str_tol) * best_kacc)))

    # F1-STR: relative stable region around best F1
    F1_STR_micro = float(np.mean(micro_f1 >= (float(f1_str_tol) * best_micro)))
    F1_STR_macro = float(np.mean(macro_f1_arr >= (float(f1_str_tol) * best_macro)))
    F1_STR_example = float(np.mean(example_f1_arr >= (float(f1_str_tol) * best_example)))

    # ---- pick theta* (single-point reporting)
    theta_star = None
    theta_star_source = None
    if theta_star_from is not None:
        val_npz = np.load(theta_star_from, allow_pickle=True)
        if theta_star_key not in val_npz:
            raise KeyError(f"theta_star_key '{theta_star_key}' not found in {theta_star_from}. keys={list(val_npz.keys())}")
        theta_star = float(val_npz[theta_star_key].item())
        theta_star_source = f"loaded:{theta_star_from}:{theta_star_key}"
    else:
        mode = theta_star_mode.lower()
        if mode == "micro_f1":
            theta_star = theta_star_micro
            theta_star_source = "picked_on_split:micro_f1"
        elif mode == "macro_f1":
            theta_star = theta_star_macro
            theta_star_source = "picked_on_split:macro_f1"
        elif mode == "example_f1":
            theta_star = theta_star_example
            theta_star_source = "picked_on_split:example_f1"
        elif mode == "kacc":
            theta_star = theta_kacc
            theta_star_source = "picked_on_split:kacc"
        elif mode == "kmae":
            theta_star = theta_kmae
            theta_star_source = "picked_on_split:kmae"
        else:
            raise ValueError("theta_star_mode must be one of: micro_f1, macro_f1, example_f1, kacc, kmae")

    # Single point metrics at theta*
    pred_star = (S >= float(theta_star))
    P_star, R_star, F1_star = micro_prf1(pred_star, mask)
    macro_star = macro_f1(pred_star, mask)
    ex_star = example_f1(pred_star, mask)

    # Pred-support uniformity at theta* using p_dist (fair/comparable)
    pred_u = pred_support_uniformity(P_dist, pred_star)

    # Debug k-hat if requested
    debug = None
    if verbose_khat:
        k_true = K.numpy()
        k_hat = pred_star.sum(dim=-1).numpy()
        unique, counts = np.unique(k_hat, return_counts=True)
        pairs = list(zip(unique.tolist(), counts.tolist()))
        pairs.sort(key=lambda x: -x[1])
        debug = {
            "k_true_mean": float(k_true.mean()),
            "k_true_std": float(k_true.std()),
            "k_true_min": int(k_true.min()),
            "k_true_max": int(k_true.max()),
            "k_hat_mean": float(k_hat.mean()),
            "k_hat_std": float(k_hat.std()),
            "k_hat_min": int(k_hat.min()),
            "k_hat_max": int(k_hat.max()),
            "k_hat_top_counts": pairs[:12],
        }

    return {
        # oracle
        **oracle,
        # cardinality + robustness
        "best_kmae": best_kmae,
        "theta_best_kmae": theta_kmae,
        "best_kacc": best_kacc,
        "theta_best_kacc": theta_kacc,
        "STR": STR,
        # F1 sweeps
        "best_micro_f1": best_micro,
        "theta_star_micro_f1": theta_star_micro,
        "best_macro_f1": best_macro,
        "theta_star_macro_f1": theta_star_macro,
        "best_example_f1": best_example,
        "theta_star_example_f1": theta_star_example,
        "F1_STR_micro": F1_STR_micro,
        "F1_STR_macro": F1_STR_macro,
        "F1_STR_example": F1_STR_example,
        # theta*
        "theta_star": float(theta_star),
        "theta_star_source": theta_star_source,
        "micro_P_star": float(P_star),
        "micro_R_star": float(R_star),
        "micro_F1_star": float(F1_star),
        "macro_F1_star": float(macro_star),
        "example_F1_star": float(ex_star),
        # pred-support uniformity
        "khat_mean_star": float(pred_u["khat_mean"]),
        "pred_mass_mean_star": float(pred_u["pred_mass_mean"]),
        "Pred_U_KL_star": float(pred_u["Pred_U_KL"]),
        "Pred_U_gap_star": float(pred_u["Pred_U_gap"]),
        # arrays
        "thetas": thetas,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1_arr,
        "example_f1": example_f1_arr,
        "kmae": kmae,
        "kacc": kacc,
        # debug
        "debug": debug,
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)

    # theta sweep
    ap.add_argument("--theta_min", type=float, default=1e-3)
    ap.add_argument("--theta_max", type=float, default=0.5)
    ap.add_argument("--theta_points", type=int, default=2001)
    ap.add_argument("--theta_logspace", action="store_true")

    # robustness
    ap.add_argument("--str_tol", type=float, default=0.95)
    ap.add_argument("--f1_str_tol", type=float, default=0.98)

    # theta* handling
    ap.add_argument("--theta_star_mode", type=str, default="kacc",
                    help="When picking theta* on the SAME split: micro_f1|macro_f1|example_f1|kacc|kmae")
    ap.add_argument("--theta_star_from", type=str, default=None,
                    help="(recommended for test) load theta* from val npz")
    ap.add_argument("--theta_star_key", type=str, default="theta_star_micro_f1",
                    help="key inside --theta_star_from npz (e.g., theta_star_micro_f1)")

    ap.add_argument("--verbose_khat", action="store_true")
    ap.add_argument("--out_npz", type=str, required=True)

    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[Device] {device}")
    print(f"[Eval-Add] data={args.data}, ckpt={args.ckpt}, split={args.split}")

    data_npz = np.load(args.data)
    d = int(data_npz["d"])
    m = int(data_npz["m"])

    ds = NpzMultilabelDataset(args.data, args.split)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    model = load_ckpt_build_model(args.ckpt, d=d, m=m, hidden=args.hidden, device=device)

    out = run_eval(
        model=model,
        loader=loader,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_points=args.theta_points,
        theta_logspace=args.theta_logspace,
        str_tol=args.str_tol,
        f1_str_tol=args.f1_str_tol,
        theta_star_mode=args.theta_star_mode,
        theta_star_from=args.theta_star_from,
        theta_star_key=args.theta_star_key,
        verbose_khat=args.verbose_khat,
        device=device,
    )

    # Pretty print
    split_name = args.split
    print(f"\n==== Metrics (Sigmoid baseline, {split_name}) ====\n")

    print("-- Coverage on TRUE support (using p_dist = normalized sigmoid for comparability) --")
    print(f"True-support mass  E[s_mass] (higher better): {out['s_mass_mean']:.6f}")
    print(f"SupportNLL  E[-log(s_mass)] (lower better): {out['support_nll']:.6f}\n")

    print("-- Uniformity on TRUE support (oracle, p_dist) --")
    print(f"U_gap_oracle (lower better): {out['U_gap_oracle']:.6f}")
    print(f"U_KL_oracle  (lower better): {out['U_KL_oracle']:.6f}\n")

    print("-- Mass-weighted uniformity on TRUE support (oracle, coverage-aware) --")
    print(f"U_gap_weighted (lower better): {out['U_gap_weighted']:.6f}")
    print(f"U_KL_weighted  (lower better): {out['U_KL_weighted']:.6f}\n")

    print("-- Cardinality / threshold robustness (sweep on NATURAL score = sigmoid prob) --")
    print(f"Best (hard) KMAE over theta: {out['best_kmae']:.6f} at theta={out['theta_best_kmae']:.8f}")
    print(f"Best (hard) KAcc over theta: {out['best_kacc']:.6f} at theta={out['theta_best_kacc']:.8f}")
    print(f"STR (tol={args.str_tol}) (higher better): {out['STR']:.6f}\n")

    print("-- Multilabel F1 sweep (sweep on NATURAL score = sigmoid prob) --")
    print(f"Best micro-F1 over theta: {out['best_micro_f1']:.6f} at theta={out['theta_star_micro_f1']:.8f}")
    print(f"Best macro-F1 over theta: {out['best_macro_f1']:.6f} at theta={out['theta_star_macro_f1']:.8f}")
    print(f"Best example-F1 over theta: {out['best_example_f1']:.6f} at theta={out['theta_star_example_f1']:.8f}")
    print(f"F1-STR_micro (tol={args.f1_str_tol}) (higher better): {out['F1_STR_micro']:.6f}")
    print(f"F1-STR_macro (tol={args.f1_str_tol}) (higher better): {out['F1_STR_macro']:.6f}")
    print(f"F1-STR_example (tol={args.f1_str_tol}) (higher better): {out['F1_STR_example']:.6f}\n")

    print("-- Single-point metrics at theta* (for reporting; theta* source matters) --")
    print(f"theta* = {out['theta_star']:.8f}  (source: {out['theta_star_source']})")
    print(f"micro-P/R/F1 @ theta*: {out['micro_P_star']:.6f} / {out['micro_R_star']:.6f} / {out['micro_F1_star']:.6f}")
    print(f"macro-F1 @ theta*: {out['macro_F1_star']:.6f}")
    print(f"example-F1 @ theta*: {out['example_F1_star']:.6f}\n")

    print("-- Pred-support uniformity at theta* (using p_dist) --")
    print(f"E[k_hat(theta*)] : {out['khat_mean_star']:.6f}")
    print(f"E[pred_mass(theta*)] : {out['pred_mass_mean_star']:.6f}")
    print(f"Pred U_KL(theta*) (lower better): {out['Pred_U_KL_star']:.6f}")
    print(f"Pred U_gap(theta*) (lower better): {out['Pred_U_gap_star']:.6f}")

    if out["debug"] is not None:
        dbg = out["debug"]
        print(f"[Debug] k_true stats: mean={dbg['k_true_mean']:.3f}, std={dbg['k_true_std']:.3f}, min={dbg['k_true_min']}, max={dbg['k_true_max']}")
        print(f"[Debug] k_hat stats @ theta: theta={out['theta_star']:.8f}, mean={dbg['k_hat_mean']:.3f}, std={dbg['k_hat_std']:.3f}, min={dbg['k_hat_min']}, max={dbg['k_hat_max']}")
        print(f"[Debug] k_hat top counts (value,count): {dbg['k_hat_top_counts']}")

    # Save NPZ (include arrays + key scalars)
    save_dict = {
        "theta_min": np.array(args.theta_min, dtype=np.float64),
        "theta_max": np.array(args.theta_max, dtype=np.float64),
        "theta_points": np.array(args.theta_points, dtype=np.int64),
        "theta_logspace": np.array(int(args.theta_logspace), dtype=np.int64),
        "str_tol": np.array(args.str_tol, dtype=np.float64),
        "f1_str_tol": np.array(args.f1_str_tol, dtype=np.float64),

        # oracle (fair: p_dist)
        "s_mass_mean": np.array(out["s_mass_mean"], dtype=np.float64),
        "support_nll": np.array(out["support_nll"], dtype=np.float64),
        "U_gap_oracle": np.array(out["U_gap_oracle"], dtype=np.float64),
        "U_KL_oracle": np.array(out["U_KL_oracle"], dtype=np.float64),
        "U_gap_weighted": np.array(out["U_gap_weighted"], dtype=np.float64),
        "U_KL_weighted": np.array(out["U_KL_weighted"], dtype=np.float64),

        # cardinality/robustness (sweep on natural score)
        "best_kmae": np.array(out["best_kmae"], dtype=np.float64),
        "theta_best_kmae": np.array(out["theta_best_kmae"], dtype=np.float64),
        "best_kacc": np.array(out["best_kacc"], dtype=np.float64),
        "theta_best_kacc": np.array(out["theta_best_kacc"], dtype=np.float64),
        "STR": np.array(out["STR"], dtype=np.float64),

        # F1 sweep (sweep on natural score)
        "best_micro_f1": np.array(out["best_micro_f1"], dtype=np.float64),
        "theta_star_micro_f1": np.array(out["theta_star_micro_f1"], dtype=np.float64),
        "best_macro_f1": np.array(out["best_macro_f1"], dtype=np.float64),
        "theta_star_macro_f1": np.array(out["theta_star_macro_f1"], dtype=np.float64),
        "best_example_f1": np.array(out["best_example_f1"], dtype=np.float64),
        "theta_star_example_f1": np.array(out["theta_star_example_f1"], dtype=np.float64),
        "F1_STR_micro": np.array(out["F1_STR_micro"], dtype=np.float64),
        "F1_STR_macro": np.array(out["F1_STR_macro"], dtype=np.float64),
        "F1_STR_example": np.array(out["F1_STR_example"], dtype=np.float64),

        # theta*
        "theta_star": np.array(out["theta_star"], dtype=np.float64),
        "theta_star_source": np.array(out["theta_star_source"]),
        "micro_P_star": np.array(out["micro_P_star"], dtype=np.float64),
        "micro_R_star": np.array(out["micro_R_star"], dtype=np.float64),
        "micro_F1_star": np.array(out["micro_F1_star"], dtype=np.float64),
        "macro_F1_star": np.array(out["macro_F1_star"], dtype=np.float64),
        "example_F1_star": np.array(out["example_F1_star"], dtype=np.float64),

        # pred uniformity @ theta* (fair: p_dist)
        "khat_mean_star": np.array(out["khat_mean_star"], dtype=np.float64),
        "pred_mass_mean_star": np.array(out["pred_mass_mean_star"], dtype=np.float64),
        "Pred_U_KL_star": np.array(out["Pred_U_KL_star"], dtype=np.float64),
        "Pred_U_gap_star": np.array(out["Pred_U_gap_star"], dtype=np.float64),

        # arrays
        "thetas": out["thetas"].astype(np.float64),
        "micro_f1": out["micro_f1"].astype(np.float64),
        "macro_f1": out["macro_f1"].astype(np.float64),
        "example_f1": out["example_f1"].astype(np.float64),
        "kmae": out["kmae"].astype(np.float64),
        "kacc": out["kacc"].astype(np.float64),
    }

    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez(args.out_npz, **save_dict)
    print(f"\nSaved: {args.out_npz}")


if __name__ == "__main__":
    main()


"""
0) Val (pick theta* on val)
--------------------------
BCE sigmoid
python3 eval_add_metrics.py \
  --data ./Synthetic.npz \
  --ckpt ./ckpt_bce.pt \
  --split val \
  --theta_min 0.001 --theta_max 0.5 --theta_points 2001 \
  --f1_str_tol 0.98 \
  --out_npz ./val_bce_widetheta.npz

ASL
python3 eval_add_metrics.py \
  --data ./Synthetic.npz \
  --ckpt ./ckpt_asl.pt \
  --split val \
  --theta_min 0.001 --theta_max 0.5 --theta_points 2001 \
  --f1_str_tol 0.98 \
  --out_npz ./val_asl_widetheta.npz

1) Test (load theta* from val; recommended)
-------------------------------------------
BCE sigmoid
python3 eval_add_metrics.py \
  --data ./Synthetic.npz \
  --ckpt ./ckpt_bce.pt \
  --split test \
  --theta_min 0.001 --theta_max 0.5 --theta_points 2001 \
  --theta_star_from ./val_bce_widetheta.npz \
  --theta_star_key theta_star_micro_f1 \
  --verbose_khat \
  --out_npz ./test_bce_widetheta.npz

ASL
python3 eval_add_metrics.py \
  --data ./Synthetic.npz \
  --ckpt ./ckpt_asl.pt \
  --split test \
  --theta_min 0.001 --theta_max 0.5 --theta_points 2001 \
  --theta_star_from ./val_asl_widetheta.npz \
  --theta_star_key theta_star_micro_f1 \
  --verbose_khat \
  --out_npz ./test_asl_widetheta.npz
"""
