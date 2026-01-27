import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Given data
# -------------------------
p_ent = np.array([0.4, 0.0, 0.0, 0.1, 0.2, 0.0, 0.3, 0.0])
p_beta = np.array([0.28, 0.0, 0.0, 0.26, 0.24, 0.0, 0.22, 0.0])
y_true = np.array([1, 0, 0, 1, 1, 0, 1, 0])

# -------------------------
# Helper functions
# -------------------------
def predict(p, t):
    return (p >= t).astype(int)

def labelwise_accuracy(y, yhat):
    return np.mean(y == yhat)

# -------------------------
# Threshold sweep
# -------------------------
ts = np.linspace(0.0001, 0.999, 500)  # 0~0.5면 변화가 잘 보임

acc_ent = [labelwise_accuracy(y_true, predict(p_ent, t)) for t in ts]
acc_beta = [labelwise_accuracy(y_true, predict(p_beta, t)) for t in ts]

# -------------------------
# Plot 1: Entmax
# -------------------------
plt.figure(figsize=(5, 4))
plt.plot(ts, acc_ent, color="#4E95D9", linewidth=2)  # 파랑
plt.xlabel("Threshold")
plt.ylabel("Label-wise Accuracy")
plt.title("Entmax: Threshold vs Accuracy")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------
# Plot 2: Beta-Entmax
# -------------------------
plt.figure(figsize=(5, 4))
plt.plot(ts, acc_beta, color="#47D45A", linewidth=2)  # 초록
plt.xlabel("Threshold")
plt.ylabel("Label-wise Accuracy")
plt.title("Beta_Entmax: Threshold vs Accuracy")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
