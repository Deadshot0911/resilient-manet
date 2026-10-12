"""Plot ablation study results."""
import matplotlib.pyplot as plt
import numpy as np

variants = ["Base", "+Bayes", "+FTL", "+MAML", "Full"]
dr_values = [64.44, 100.0, 100.0, 100.0, 100.0]
fpr_values = [4.62, 1.10, 1.10, 1.10, 1.10]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = np.arange(len(variants))

axes[0].bar(x, dr_values, color="seagreen")
axes[0].set_xticks(x); axes[0].set_xticklabels(variants)
axes[0].set_ylabel("Detection Rate (%)")
axes[0].set_title("Ablation: Detection Rate vs Components")
axes[0].set_ylim(0, 110)
for i, v in enumerate(dr_values):
    axes[0].text(i, v + 2, f"{v:.1f}", ha="center", fontsize=10)

axes[1].bar(x, fpr_values, color="coral")
axes[1].set_xticks(x); axes[1].set_xticklabels(variants)
axes[1].set_ylabel("FPR (%)")
axes[1].set_title("Ablation: False Positive Rate vs Components")
axes[1].set_ylim(0, max(6, max(fpr_values) * 1.3))
for i, v in enumerate(fpr_values):
    axes[1].text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("outputs/figures/ablation.png", dpi=150)
print("Saved outputs/figures/ablation.png")
