"""Phase 3: Bayesian Trust Calibration."""
import numpy as np
from config import BETA_ALPHA_0, BETA_BETA_0, DECAY_LAMBDA, RISK_K

class BayesianTrust:
    def __init__(self, num_nodes):
        self.alpha = np.full(num_nodes, BETA_ALPHA_0)
        self.beta = np.full(num_nodes, BETA_BETA_0)
        self.alpha0 = BETA_ALPHA_0
        self.beta0 = BETA_BETA_0

    def update(self, node_id, success):
        a, b = self.alpha[node_id], self.beta[node_id]
        a = self.alpha0 + DECAY_LAMBDA * (a - self.alpha0)
        b = self.beta0 + DECAY_LAMBDA * (b - self.beta0)
        if success:
            a += 1
        else:
            b += 1
        self.alpha[node_id], self.beta[node_id] = a, b

    def mean(self, node_id):
        return self.alpha[node_id] / (self.alpha[node_id] + self.beta[node_id])

    def std(self, node_id):
        a, b = self.alpha[node_id], self.beta[node_id]
        n = a + b
        return np.sqrt(a * b / (n ** 2 * (n + 1)))

    def risk_adjusted_score(self, node_id, k=RISK_K):
        return self.mean(node_id) - k * self.std(node_id)
