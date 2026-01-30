# visualize_top10.py
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ✅ 너의 train_Synthetic.py에서 그대로 가져옴
from train_Synthetic import SyntheticDataset, MLPLogits, BetaHead, beta_entmax

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

@torch.no_grad()
def infer_probs_on_test(npz_path: str, ckpt_path: str, batch: int, fixed_beta: float):
    device = get_device()
    data = np.load(npz_path)
    d = int(data["d"])
    m = int(data["m"])

    ds_test = SyntheticDataset(npz_path, "test")
    loader = DataLoader(ds_test, batch_size=batch, shuffle=False, num_workers=0)

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

    all_p, all_y, all_k = [], [], []
    for x, y, k in loader:
        x = x.to(device)
        z = model_g(x)

        if beta_head is None:
            beta = torch.full((x.size(0), 1), float(fixed_beta), device=device, dtype=z.dtype)
        else:
            beta = beta_head(x)

        p = beta_entmax(z, beta, alpha=1.5, dim=-1, n_iter=30)
        p = safe_prob_eval(p)

        all_p.append(p.cpu())
        all_y.append(y)
        all_k.append(k)

    P = torch.cat(all_p, dim=0).numpy()  # [N,m]
    Y = torch.cat(all_y, dim=0).numpy()  # [N,m]
    K = torch.cat(all_k, dim=0).numpy()  # [N]
    return P, Y, K

def plot_topk(P, Y, K, indices, topk: int = 10, save_prefix: str = None):
    for idx in indices:
        p = P[idx]  # [m]
        y = Y[idx]  # [m]
        k_true = int(K[idx])
        true_pos = np.where(y > 0.5)[0].tolist()

        top_idx = np.argsort(-p)[:topk]
        top_p = p[top_idx]
        top_y = y[top_idx]

        # top-k 안에 실제 label 몇 개 들어갔는지
        hit = int((top_y > 0.5).sum())
        miss = k_true - hit

        plt.figure(figsize=(10, 4))
        bars = plt.bar(np.arange(topk), top_p)

        # ✅ 실제 label(y=1)인 bar 위에 표시
        for i in range(topk):
            if top_y[i] > 0.5:
                plt.text(i, top_p[i], "★", ha="center", va="bottom")

        plt.xticks(np.arange(topk), [str(int(j)) for j in top_idx], rotation=0)
        plt.ylim(0, max(1e-6, float(top_p.max()) * 1.15))

        plt.title(
            f"Test sample #{idx} | k_true={k_true} | top{topk} hit={hit} (miss={miss})\n"
            f"Top{topk} label indices (★ means y=1)"
        )
        plt.xlabel("label index (top-k)")
        plt.ylabel("pred prob")

        # 참고로 true label 전체도 출력(텍스트)
        plt.figtext(0.01, -0.02, f"true labels: {true_pos}", wrap=True, ha="left", fontsize=9)

        plt.tight_layout()

        if save_prefix:
            out = f"{save_prefix}_idx{idx}_top{topk}.png"
            plt.savefig(out, dpi=160, bbox_inches="tight")
            print(f"[Saved] {out}")
            plt.close()
        else:
            plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Synthetic.npz path")
    ap.add_argument("--ckpt", type=str, required=True, help="ckpt_beta_abl2_nocard.pt path")
    ap.add_argument("--fixed_beta", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--topk", type=int, default=10)

    # 샘플 선택 방식 1) 특정 인덱스들 지정
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="e.g., --idx 0 1 2")

    # 샘플 선택 방식 2) 랜덤 N개
    ap.add_argument("--random_n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    # 저장 옵션
    ap.add_argument("--save_prefix", type=str, default=None, help="e.g., ./viz/abl2")

    args = ap.parse_args()

    P, Y, K = infer_probs_on_test(args.data, args.ckpt, args.batch, args.fixed_beta)
    N = P.shape[0]

    if args.idx is not None and len(args.idx) > 0:
        indices = [i for i in args.idx if 0 <= i < N]
    else:
        if args.random_n <= 0:
            indices = [0]  # default: 첫 샘플 1개
        else:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(N, size=min(args.random_n, N), replace=False).tolist()

    print(f"[Info] plotting indices: {indices}")
    plot_topk(P, Y, K, indices, topk=args.topk, save_prefix=args.save_prefix)

if __name__ == "__main__":
    main()

"""
# (1) test 샘플 3개를 랜덤으로 top10 시각화 (화면에 띄움)
python3 visualize_top10.py \
  --data ./Synthetic.npz \
  --ckpt ./ckpt_beta_abl2_nocard.pt \
  --random_n 3 --seed 0 --topk 10

# (2) 특정 test index만 찍기 
python3 visualize_top10.py \
  --data ./Synthetic.npz \
  --ckpt ./ckpt_beta_abl2_nocard.pt \
  --idx 3 --topk 10

python3 visualize_top10.py \
  --data ./Synthetic.npz \
  --ckpt ./ckpt_entmax15.pt \
  --idx 3 --topk 10

# (3) 파일로 저장 (예: ./viz/abl2_idx0_top10.png ...)
mkdir -p viz
python3 visualize_top10.py \
  --data ./Synthetic.npz \
  --ckpt ./ckpt_beta_abl2_nocard.pt \
  --random_n 5 --seed 1 --topk 10 \
  --save_prefix ./viz/abl2
"""