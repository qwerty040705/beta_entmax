import os
import numpy as np
from dataclasses import dataclass
from typing import Optional


# -----------------------------
# Defaults
# -----------------------------
@dataclass
class Defaults:
    total_samples: int = 50000
    m: int = 1000
    d: int = 256
    Kmax: int = 10
    lam: float = 3.0
    sigma: float = 0.3
    out_file: str = "Synthetic.npz"


# -----------------------------
# Input helpers
# -----------------------------
def ask_int(prompt: str, default: int, min_value: Optional[int] = None) -> int:
    s = input(f"{prompt} (default={default}): ").strip()
    v = default if s == "" else int(s)
    if min_value is not None and v < min_value:
        raise ValueError(f"{prompt} must be >= {min_value}")
    return v


def ask_float(prompt: str, default: float, min_value: Optional[float] = None) -> float:
    s = input(f"{prompt} (default={default}): ").strip()
    v = default if s == "" else float(s)
    if min_value is not None and v < min_value:
        raise ValueError(f"{prompt} must be >= {min_value}")
    return v


def ask_str(prompt: str, default: str) -> str:
    s = input(f"{prompt} (default={default}): ").strip()
    return default if s == "" else s


# -----------------------------
# Synthetic generation
# -----------------------------
def truncated_poisson(rng, lam, N, Kmax):
    k = rng.poisson(lam, size=N)
    return np.clip(k, 1, Kmax).astype(np.int64)


def main():
    defaults = Defaults()

    N = ask_int("How many total_samples?", defaults.total_samples, 1)
    m = ask_int("How many m?", defaults.m, 2)
    d = ask_int("How many d?", defaults.d, 1)
    Kmax = ask_int("How many Kmax?", defaults.Kmax, 1)
    lam = ask_float("How many lam?", defaults.lam, 0.0)
    sigma = ask_float("How many sigma?", defaults.sigma, 0.0)
    out_file = ask_str("How many out_file?", defaults.out_file)

    rng = np.random.default_rng()

    if Kmax > m:
        print(f"[WARN] Kmax({Kmax}) > m({m}); effective k limited by m.")

    print("\nGenerating label prototypes A ~ N(0, I_d)")
    A = rng.normal(0.0, 1.0, size=(m, d)).astype(np.float32)

    print("Generating samples...")
    k = truncated_poisson(rng, lam, N, Kmax)

    X = np.zeros((N, d), dtype=np.float32)
    y = np.zeros((N, m), dtype=np.float32)
    q = np.zeros((N, m), dtype=np.float32)
    S_idx = -np.ones((N, Kmax), dtype=np.int64)

    for n in range(N):
        kn = k[n]
        S = rng.choice(m, size=kn, replace=False)

        S_idx[n, :kn] = S
        y[n, S] = 1.0
        q[n, S] = 1.0 / kn

        mean_vec = A[S].mean(axis=0)
        noise = rng.normal(0.0, sigma, size=d)
        X[n] = mean_vec + noise

    print(f"Saving to {out_file}")
    np.savez_compressed(
        out_file,
        X=X,
        y=y,
        q=q,
        k=k,
        S_idx=S_idx,
        A=A,
        m=np.int64(m),
        d=np.int64(d),
        Kmax=np.int64(Kmax),
        lam=np.float64(lam),
        sigma=np.float64(sigma),
        N=np.int64(N),
    )

    print("[Done]")
    print("Saved keys: X, y, q, k, S_idx, A")


if __name__ == "__main__":
    main()
