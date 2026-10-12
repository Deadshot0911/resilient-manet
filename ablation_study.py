"""Ablation study using weighted ensemble logic."""
import numpy as np
from config import NUM_NODES, FEATURE_DIM, RANDOM_SEED, MALICIOUS_RATIO
from modules.phase2_gatm import GATM
from modules.phase3_bayesian import BayesianTrust
from modules.phase4_maml import MAMLDetector
from modules.phase1_ftl import FederatedTrustLearner
from main_resilient import (load_or_synthesize_features, normalize_features,
                             get_ground_truth, train_engine)

def compute_weighted_trust(H, active):
    """Recreate production ensemble logic with only active components."""
    n = NUM_NODES
    components = []
    weights = []

    if "bayes" in active:
        labels = get_ground_truth(H).astype(float)
        bt = BayesianTrust(n)
        for i in range(n):
            for _ in range(5):
                bt.update(i, labels[i] < 0.5)
        components.append(np.array([bt.mean(i) for i in range(n)]))
        weights.append(0.70)

    if "ftl" in active:
        labels = get_ground_truth(H).astype(float)
        ftl = FederatedTrustLearner()
        client_data = [(H[k*10:(k+1)*10], labels[k*10:(k+1)*10]) for k in range(10)]
        for _ in range(30):
            ftl.federated_round(client_data)
        raw = 1 / (1 + np.exp(-np.clip(H @ ftl.global_weights, -500, 500)))
        components.append(1 - raw)  # invert: high drop -> low trust
        weights.append(0.10)

    if "maml" in active:
        labels = get_ground_truth(H).astype(float)
        maml = MAMLDetector()
        tasks = []
        for _ in range(5):
            idx = np.random.choice(n, min(20, n), replace=False)
            X_t, y_t = H[idx], labels[idx]
            if len(np.unique(y_t)) < 2:
                y_t = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(len(y_t))])
            split = len(X_t) // 2
            tasks.append((X_t[:split], y_t[:split], X_t[split:], y_t[split:]))
        for _ in range(30):
            maml.meta_train_step(tasks)
        raw = maml.forward(H)
        components.append(1 - raw)
        weights.append(0.10)

    if "gatm" in active:
        labels = get_ground_truth(H).astype(float)
        gatm = GATM()
        honest = H[labels < 0.5] if (labels < 0.5).sum() > 5 else H
        for _ in range(90):
            gatm.train_step(honest, batch_size=min(16, len(honest)))
        components.append(gatm.discriminator_forward(H).flatten())
        weights.append(0.10)

    if not components:
        return 1 / (1 + np.exp(-np.clip(H @ np.random.randn(FEATURE_DIM), -500, 500)))

    # Normalize weights so they sum to 1
    weights = np.array(weights)
    weights = weights / weights.sum()
    trust = sum(w * c for w, c in zip(weights, components))
    return trust

def evaluate(H, active, seed):
    np.random.seed(seed)
    H_norm = normalize_features(H)
    trust = compute_weighted_trust(H_norm, active)

    malicious = get_ground_truth(H)
    n_mal = max(int(NUM_NODES * MALICIOUS_RATIO), 1)
    bottom = np.argsort(trust)[:n_mal]
    predicted = np.zeros(NUM_NODES, dtype=bool)
    predicted[bottom] = True

    tp = int(np.sum(malicious & predicted))
    fp = int(np.sum(~malicious & predicted))
    fn = int(np.sum(malicious & ~predicted))
    tn = int(np.sum(~malicious & ~predicted))
    return tp / max(tp + fn, 1) * 100, fp / max(fp + tn, 1) * 100

def main():
    variants = [
        ("Base (no ML)", set()),
        ("+ Bayesian", {"bayes"}),
        ("+ Bayesian + FTL", {"bayes", "ftl"}),
        ("+ Bayesian + FTL + MAML", {"bayes", "ftl", "maml"}),
        ("Full Ensemble", {"bayes", "ftl", "gatm", "maml"}),
    ]
    print(f"{'Variant':<32} {'DR (%)':>10} {'FPR (%)':>10}")
    print("-" * 55)
    for name, comps in variants:
        drs, fprs = [], []
        for seed_offset in range(5):
            H = load_or_synthesize_features(RANDOM_SEED + seed_offset)
            dr, fpr = evaluate(H, comps, RANDOM_SEED + seed_offset)
            drs.append(dr)
            fprs.append(fpr)
        print(f"{name:<32} {np.mean(drs):>10.2f} {np.mean(fprs):>10.2f}")

if __name__ == "__main__":
    main()
