"""Plot RESILIENT-MANET results."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("outputs/resilient_manet_final_benchmark.json") as f:
    data = json.load(f)

runs = list(range(1, len(data["runs"]) + 1))
dr = [r["detection_rate"] for r in data["runs"]]
fpr = [r["fpr"] for r in data["runs"]]
thresholds = [r["threshold"] for r in data["runs"]]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(runs, dr, color="seagreen")
axes[0].set_xlabel("Monte Carlo Run")
axes[0].set_ylabel("Detection Rate (%)")
axes[0].set_title(f"Detection Rate (mean = {data['mean_detection_rate']:.1f}%)")
axes[0].set_ylim(0, 105)
axes[0].axhline(100, color="gray", linestyle="--", alpha=0.5)

axes[1].bar(runs, fpr, color="coral")
axes[1].set_xlabel("Monte Carlo Run")
axes[1].set_ylabel("False Positive Rate (%)")
axes[1].set_title(f"FPR (mean = {data['mean_fpr']:.2f}%)")
axes[1].set_ylim(0, max(5, max(fpr) * 1.5 + 0.5))

axes[2].bar(runs, thresholds, color="steelblue")
axes[2].set_xlabel("Monte Carlo Run")
axes[2].set_ylabel("Auto-detected Threshold")
axes[2].set_title("Threshold Per Run")
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("outputs/figures/results.png", dpi=150)
print("Saved outputs/figures/results.png")
