import numpy as np

path = "/Users/dnbn/code/beta_entmax/Synthetic/Synthetic.npz"

data = np.load(path)

print("==== Keys ====")
for k in data.files:
    print(k)

print("\n==== Shapes ====")
for k in ["X", "y", "q", "k", "S_idx", "A"]:
    if k in data:
        print(f"{k}: {data[k].shape}")

# -----------------------------
# Basic sanity checks
# -----------------------------
X = data["X"]
y = data["y"]
q = data["q"]
k = data["k"]
S_idx = data["S_idx"]
A = data["A"]

N, d = X.shape
m = y.shape[1]

print("\n==== Basic stats ====")
print(f"N = {N}, m = {m}, d = {d}")
print(f"k mean = {k.mean():.3f}, min = {k.min()}, max = {k.max()}")

# -----------------------------
# Check a few samples
# -----------------------------
def inspect_sample(n: int):
    print(f"\n---- Sample {n} ----")
    kn = k[n]
    S = S_idx[n][:kn]

    print("k =", kn)
    print("labels S =", S)

    print("y[S] =", y[n, S])
    print("q[S] =", q[n, S], " sum(q[S]) =", q[n, S].sum())

    # Check x construction
    x = X[n]
    mean_proto = A[S].mean(axis=0)

    diff = np.linalg.norm(x - mean_proto)
    print("||x - mean(A[S])|| =", diff)

inspect_sample(0)
inspect_sample(1)
inspect_sample(2)

# -----------------------------
# Distribution checks
# -----------------------------
print("\n==== Cardinality distribution ====")
unique_k, counts = np.unique(k, return_counts=True)
for uk, c in zip(unique_k, counts):
    print(f"k = {uk}: {c} samples ({c/N*100:.2f}%)")

print("\n==== Uniformity check (q) ====")
max_err = 0.0
for n in range(10):
    kn = k[n]
    S = S_idx[n][:kn]
    err = np.abs(q[n, S] - 1.0/kn).max()
    max_err = max(max_err, err)
print("max |q - 1/k| over first 10 samples =", max_err)
