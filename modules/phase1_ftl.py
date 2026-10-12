"""Phase 1: Federated Trust Learning with Differential Privacy."""
import numpy as np
from config import (NUM_CLUSTER_HEADS, DP_EPSILON, DP_DELTA,
                    L2_CLIP_NORM, FEATURE_DIM)

def add_dp_noise(gradient, epsilon=DP_EPSILON, delta=DP_DELTA, clip=L2_CLIP_NORM):
    norm = np.linalg.norm(gradient)
    if norm > clip:
        gradient = gradient * (clip / norm)
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * clip / epsilon
    return gradient + np.random.normal(0, sigma, gradient.shape)

class FederatedTrustLearner:
    def __init__(self, num_heads=NUM_CLUSTER_HEADS, feature_dim=FEATURE_DIM):
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.global_weights = np.zeros(feature_dim)
        self.reputation = np.ones(num_heads)

    def local_train(self, X, y, epochs=5, lr=0.01):
        w = self.global_weights.copy()
        n = max(len(y), 1)
        for _ in range(epochs):
            preds = 1 / (1 + np.exp(-np.clip(X @ w, -500, 500)))
            grad = X.T @ (preds - y) / n
            w -= lr * grad
        return w - self.global_weights

    def reputation_weighted_fedavg(self, gradients, data_sizes):
        weights = np.array([data_sizes[k] * self.reputation[k]
                            for k in range(len(gradients))])
        weights = weights / weights.sum()
        return sum(weights[k] * gradients[k] for k in range(len(gradients)))

    def federated_round(self, client_data):
        gradients, sizes = [], []
        for X, y in client_data:
            grad = self.local_train(X, y)
            gradients.append(add_dp_noise(grad))
            sizes.append(len(y))
        avg_grad = self.reputation_weighted_fedavg(gradients, sizes)
        self.global_weights += avg_grad
        return self.global_weights
