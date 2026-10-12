"""Inject malicious-node patterns into real NS-3 feature matrix."""
import numpy as np

# Load the real NS-3 features
H_real = np.load("outputs/node_features.npy").copy()
num_nodes = H_real.shape[0]
n_malicious = 10

# Pick 10 random nodes to become "malicious"
np.random.seed(42)
malicious = np.random.choice(num_nodes, n_malicious, replace=False)

# Attack patterns
attack_types = {
    "blackhole": {"drop_ratio": (0.95, 1.00), "forward_ratio": (0.00, 0.05)},
    "grayhole":  {"drop_ratio": (0.55, 0.70), "forward_ratio": (0.30, 0.45)},
    "on_off":    {"drop_ratio": (0.40, 0.60), "forward_ratio": (0.40, 0.60)},
    "collusion": {"drop_ratio": (0.70, 0.85), "forward_ratio": (0.15, 0.30)},
    "zero_day":  {"drop_ratio": (0.50, 0.75), "forward_ratio": (0.25, 0.50)},
}
type_keys = list(attack_types.keys())

for i, node_idx in enumerate(malicious):
    atk = attack_types[type_keys[i % len(type_keys)]]
    dr = np.random.uniform(*atk["drop_ratio"])
    fr = np.random.uniform(*atk["forward_ratio"])
    H_real[node_idx, 3] = dr      # drop ratio
    H_real[node_idx, 7] = fr      # forwarding ratio
    H_real[node_idx, 8] = fr      # trust proxy

np.save("outputs/node_features.npy", H_real)
print(f"Injected {n_malicious} malicious nodes: {sorted(malicious.tolist())}")
print(f"Attack types used: {[type_keys[i % len(type_keys)] for i in range(n_malicious)]}")
