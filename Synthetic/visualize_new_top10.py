# visualize_new_top10.py
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ✅ 프로젝트에 이미 있는 train_Synthetic.py에서 그대로 import
from train_Synthetic import SyntheticDataset, MLPLogits, BetaHead, beta_entmax


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
    """
    Eval-safe probability normalization:
      1) nan/inf -> 0
      2) clamp negatives to 0
      3) renorm to sum=1
    """
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = p.clamp(min=0.0)
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    return p


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def infer_probs_on_split(
    npz_path: str,
    ckpt_path: str,
    split: str,
    batch: int,
    fixed_beta: float,
):
    device = get_device()
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
    if ckpt.get("beta_head") is not None:
        beta_head = BetaHead(
            d=d,
            hidden=256,
            beta_min=0.0,
            beta_max=float(args.get("beta_max", 10.0)),
        ).to(device)
        beta_head.load_state_dict(ckpt["beta_head"])
        beta_head.eval()

    all_p, all_y, all_k, all_beta = [], [], [], []

    for x, y, k in loader:
        x = x.to(device)
        z = model_g(x)

        if beta_head is None:
            beta = torch.full((x.size(0), 1), float(fixed_beta), device=device, dtype=z.dtype)
        else:
            beta = beta_head(x)

        p = beta_entmax(z, beta, alpha=1.5, dim=-1, n_iter=30)
        p = safe_prob_eval(p)

        all_p.append(p.detach().cpu())
        all_y.append(y.detach().cpu())
        all_k.append(k.detach().cpu())
        all_beta.append(beta.detach().cpu())

    P = torch.cat(all_p, dim=0).numpy()      # [N,m]
    Y = torch.cat(all_y, dim=0).numpy()      # [N,m]
    K = torch.cat(all_k, dim=0).numpy()      # [N]
    B = torch.cat(all_beta, dim=0).numpy()   # [N,1]

    return P, Y, K, B


# ============================================================
# Plotting
# ============================================================

def _ensure_dir_for_prefix(save_prefix: str):
    d = os.path.dirname(save_prefix)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def plot_topk(
    P: np.ndarray,
    Y: np.ndarray,
    K: np.ndarray,
    B: np.ndarray,
    indices,
    topk: int = 10,
    theta: float = None,
    save_prefix: str = None,
    show_beta: bool = True,
):
    """
    For each sample:
      - Plot top-k predicted probabilities.
      - Mark true labels (y=1) with ★.
      - Print:
          * k_true
          * top-k hit/miss
          * (optional) predicted set at theta: k_hat, hit/miss
          * beta value
    """
    for idx in indices:
        p = P[idx]  # [m]
        y = Y[idx]  # [m]
        k_true = int(K[idx])
        beta = float(B[idx].reshape(-1)[0])

        true_pos = np.where(y > 0.5)[0]
        true_set = set(true_pos.tolist())

        top_idx = np.argsort(-p)[:topk]
        top_p = p[top_idx]
        top_y = y[top_idx]
        hit_topk = int((top_y > 0.5).sum())
        miss_topk = max(k_true - hit_topk, 0)

        # thresholded predicted set (optional)
        pred_info = ""
        if theta is not None:
            pred_mask = (p >= float(theta))
            pred_idx = np.where(pred_mask)[0]
            pred_set = set(pred_idx.tolist())
            k_hat = int(len(pred_idx))
            hit_pred = int(len(pred_set.intersection(true_set)))
            miss_pred = max(k_true - hit_pred, 0)
            fp_pred = max(k_hat - hit_pred, 0)
            pred_mass = float((p * pred_mask).sum())
            pred_info = (
                f"\nθ={theta:g} | k_hat={k_hat} | hit={hit_pred} | miss={miss_pred} | FP={fp_pred} | pred_mass={pred_mass:.4f}"
            )

        plt.figure(figsize=(11, 4))
        plt.bar(np.arange(topk), top_p)

        for i in range(topk):
            if top_y[i] > 0.5:
                plt.text(i, top_p[i], "★", ha="center", va="bottom", fontsize=12)

        plt.xticks(np.arange(topk), [str(int(j)) for j in top_idx], rotation=0)
        plt.ylim(0, max(1e-6, float(top_p.max()) * 1.15))

        title = f"{idx} | k_true={k_true} | top{topk} hit={hit_topk} (miss={miss_topk})"
        if show_beta:
            title += f" | beta={beta:.4f}"
        title += pred_info

        plt.title(title)
        plt.xlabel("label index (top-k)")
        plt.ylabel("pred prob")

        # print all true labels under figure
        plt.figtext(0.01, -0.02, f"true labels: {sorted(list(true_set))}", wrap=True, ha="left", fontsize=9)
        plt.tight_layout()

        if save_prefix:
            _ensure_dir_for_prefix(save_prefix)
            out = f"{save_prefix}_idx{idx}_top{topk}.png"
            plt.savefig(out, dpi=160, bbox_inches="tight")
            print(f"[Saved] {out}")
            plt.close()
        else:
            plt.show()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, required=True, help="Synthetic.npz path")
    ap.add_argument(
        "--ckpt",
        type=str,
        default="/Users/dnbn/code/beta_entmax/Synthetic/ckpt_beta_new.pt",
        help="checkpoint path (default: ckpt_beta_new.pt)",
    )

    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--fixed_beta", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--topk", type=int, default=10)

    # 선택 방식: 특정 idx or 랜덤 N개
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="e.g., --idx 0 1 2")
    ap.add_argument("--random_n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    # (NEW) theta를 주면 predicted-set 정보도 같이 찍음
    ap.add_argument("--theta", type=float, default=None, help="If set, also report k_hat at threshold theta")

    # 저장 옵션
    ap.add_argument("--save_prefix", type=str, default=None, help="e.g., ./viz/new_beta")
    ap.add_argument("--no_beta_in_title", action="store_true", help="Do not show beta in plot title")

    args = ap.parse_args()

    P, Y, K, B = infer_probs_on_split(args.data, args.ckpt, args.split, args.batch, args.fixed_beta)

    # Beta stats (전체)
    b_flat = B.reshape(-1)
    print(f"[Info] split={args.split} | N={P.shape[0]} | m={P.shape[1]}")
    print(f"[Info] beta mean/std: {float(b_flat.mean()):.6f} / {float(b_flat.std()):.6f}")

    N = P.shape[0]
    if args.idx is not None and len(args.idx) > 0:
        indices = [i for i in args.idx if 0 <= i < N]
    else:
        if args.random_n <= 0:
            indices = [0]
        else:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(N, size=min(args.random_n, N), replace=False).tolist()

    print(f"[Info] plotting indices: {indices}")
    plot_topk(
        P, Y, K, B,
        indices=indices,
        topk=args.topk,
        theta=args.theta,
        save_prefix=args.save_prefix,
        show_beta=(not args.no_beta_in_title),
    )


if __name__ == "__main__":
    main()

"""
# ============================================================
# Usage examples (ckpt_beta_new.pt default)
# ============================================================

cd Synthetic

# (1) test에서 랜덤 5개 top10, 화면 표시
python3 visualize_new_top10.py \
  --data ./Synthetic.npz \
  --split test \
  --random_n 5 --seed 0 --topk 10

# (2) test에서 특정 index만 top10
python3 visualize_new_top10.py \
  --data ./Synthetic.npz \
  --split test \
  --idx 3 10 42 --topk 10

# (3) val에서 theta* 기준 predicted-set(k_hat)도 같이 보고 싶으면
#     (예: 너가 찾은 theta* ~ 0.041918)
python3 visualize_new_top10.py \
  --data ./Synthetic.npz \
  --split test \
  --theta 0.041918 \
  --random_n 5 --seed 1 --topk 10

# (4) 파일로 저장
mkdir -p viz
python3 visualize_new_top10.py \
  --data ./Synthetic.npz \
  --split test \
  --theta 0.041918 \
  --random_n 10 --seed 2 --topk 10 \
  --save_prefix ./viz/beta_new
"""
