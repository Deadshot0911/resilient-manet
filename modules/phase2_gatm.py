"""Phase 2: Generative Adversarial Trust Model (GATM)."""
import numpy as np
from config import LATENT_DIM, HIDDEN_DIM, FEATURE_DIM, GAN_LR

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def relu_grad(x): return (x > 0).astype(float)

class GATM:
    def __init__(self):
        self.G_W1 = np.random.randn(LATENT_DIM, HIDDEN_DIM) * 0.1
        self.G_b1 = np.zeros(HIDDEN_DIM)
        self.G_W2 = np.random.randn(HIDDEN_DIM, FEATURE_DIM) * 0.1
        self.G_b2 = np.zeros(FEATURE_DIM)
        self.D_W1 = np.random.randn(FEATURE_DIM, HIDDEN_DIM) * 0.1
        self.D_b1 = np.zeros(HIDDEN_DIM)
        self.D_W2 = np.random.randn(HIDDEN_DIM, 1) * 0.1
        self.D_b2 = np.zeros(1)

    def generator_forward(self, z):
        h = relu(z @ self.G_W1 + self.G_b1)
        return sigmoid(h @ self.G_W2 + self.G_b2)

    def discriminator_forward(self, x):
        h = relu(x @ self.D_W1 + self.D_b1)
        return sigmoid(h @ self.D_W2 + self.D_b2)

    def train_step(self, real_features, batch_size=32):
        idx = np.random.choice(len(real_features),
                               min(batch_size, len(real_features)), replace=False)
        real = real_features[idx]
        z = np.random.randn(len(real), LATENT_DIM)
        fake = self.generator_forward(z)
        d_real = self.discriminator_forward(real)
        d_fake = self.discriminator_forward(fake)
        d_loss_real = (d_real - 1) / len(real)
        d_loss_fake = d_fake / len(real)
        h_real = relu(real @ self.D_W1 + self.D_b1)
        h_fake = relu(fake @ self.D_W1 + self.D_b1)
        self.D_W2 -= GAN_LR * (h_real.T @ d_loss_real + h_fake.T @ d_loss_fake)
        self.D_b2 -= GAN_LR * (d_loss_real.sum() + d_loss_fake.sum())
        grad_h_real = (d_loss_real @ self.D_W2.T) * relu_grad(h_real)
        grad_h_fake = (d_loss_fake @ self.D_W2.T) * relu_grad(h_fake)
        self.D_W1 -= GAN_LR * (real.T @ grad_h_real + fake.T @ grad_h_fake)
        self.D_b1 -= GAN_LR * (grad_h_real.sum(0) + grad_h_fake.sum(0))
        z = np.random.randn(len(real), LATENT_DIM)
        fake = self.generator_forward(z)
        d_fake = self.discriminator_forward(fake)
        g_loss = (d_fake - 1) / len(real)
        h_fake = relu(fake @ self.D_W1 + self.D_b1)
        grad_fake = (g_loss @ self.D_W2.T) * relu_grad(h_fake)
        grad_fake = grad_fake @ self.D_W1.T
        h_gen = relu(z @ self.G_W1 + self.G_b1)
        grad_h_gen = ((grad_fake * fake * (1 - fake)) @ self.G_W2.T) * relu_grad(h_gen)
        self.G_W2 -= GAN_LR * (h_gen.T @ (grad_fake * fake * (1 - fake)))
        self.G_b2 -= GAN_LR * (grad_fake * fake * (1 - fake)).sum(0)
        self.G_W1 -= GAN_LR * (z.T @ grad_h_gen)
        self.G_b1 -= GAN_LR * grad_h_gen.sum(0)

    def synthesize_attacks(self, n=100):
        z = np.random.randn(n, LATENT_DIM)
        return self.generator_forward(z)
