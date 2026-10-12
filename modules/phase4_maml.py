"""Phase 4: MAML meta-learning detector."""
import numpy as np
from config import (MAML_INNER_LR, MAML_OUTER_LR,
                    MAML_INNER_STEPS, FEATURE_DIM)

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class MAMLDetector:
    def __init__(self, feature_dim=FEATURE_DIM):
        self.theta = np.random.randn(feature_dim) * 0.01

    def forward(self, X, theta=None):
        if theta is None:
            theta = self.theta
        return sigmoid(X @ theta)

    def loss(self, X, y, theta=None):
        preds = self.forward(X, theta)
        return -np.mean(y * np.log(preds + 1e-8) +
                        (1 - y) * np.log(1 - preds + 1e-8))

    def inner_update(self, X_s, y_s, theta=None):
        if theta is None:
            theta = self.theta.copy()
        for _ in range(MAML_INNER_STEPS):
            preds = self.forward(X_s, theta)
            grad = X_s.T @ (preds - y_s) / max(len(y_s), 1)
            theta = theta - MAML_INNER_LR * grad
        return theta

    def meta_train_step(self, tasks):
        meta_grad = np.zeros_like(self.theta)
        for X_s, y_s, X_q, y_q in tasks:
            adapted = self.inner_update(X_s, y_s)
            preds_q = self.forward(X_q, adapted)
            grad_q = X_q.T @ (preds_q - y_q) / max(len(y_q), 1)
            meta_grad += grad_q
        meta_grad /= max(len(tasks), 1)
        self.theta -= MAML_OUTER_LR * meta_grad

    def adapt_few_shot(self, X_s, y_s):
        return self.inner_update(X_s, y_s)

    def predict(self, X, theta=None):
        return (self.forward(X, theta) > 0.5).astype(int)
