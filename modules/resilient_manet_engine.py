"""Phase 5: Trust ensemble + HLOA routing (fixed signs)."""
import numpy as np
from config import (FEATURE_DIM, HLOA_ITERATIONS, HLOA_POPULATION,
                    CONFIDENCE_THRESHOLD)
from modules.phase1_ftl import FederatedTrustLearner
from modules.phase2_gatm import GATM
from modules.phase3_bayesian import BayesianTrust
from modules.phase4_maml import MAMLDetector

class ResilientManetEngine:
    def __init__(self, num_nodes=100):
        self.num_nodes = num_nodes
        self.ftl = FederatedTrustLearner()
        self.gatm = GATM()
        self.bayes = BayesianTrust(num_nodes)
        self.maml = MAMLDetector()
        np.random.seed(42)
        self.W_edge = np.random.randn(FEATURE_DIM) * 0.1

    def compute_ensemble_trust(self, H):
        """
        Compute trust score per node.

        HIGH trust  = likely honest
        LOW trust   = likely malicious
        """
        n = self.num_nodes

        # 1. Bayesian: we calibrated it with success=True for honest, False for malicious
        # so bt.mean() is high for honest -> this is already correctly oriented
        T_bayes = np.array([self.bayes.mean(i) for i in range(n)])

        # 2. FTL: trained logistic regression on labels where y=1 means HONEST
        # (we set y = (H[:,3] > 0.5).astype(float) then inverted?
        # Let's check: in main_resilient.py train_engine uses
        #   labels = get_ground_truth(H).astype(float)
        # where get_ground_truth returns H[:,3] > 0.5 = True for MALICIOUS.
        # So FTL learned: high drop ratio -> label 1 -> "malicious"
        # We need to INVERT this to get trust.
        T_ftl_raw = 1 / (1 + np.exp(-np.clip(H @ self.ftl.global_weights, -500, 500)))
        T_ftl = 1 - T_ftl_raw  # invert: high drop_ratio -> low trust

        # 3. GATM: discriminator trained to say 1 for honest features
        # Output is "looks honest" probability, already correctly oriented
        T_gatm = self.gatm.discriminator_forward(H).flatten()

        # 4. MAML: trained same as FTL, so labels are "malicious"
        T_maml_raw = self.maml.forward(H)
        T_maml = 1 - T_maml_raw  # invert

        # Weighted ensemble
        T = (0.70 * T_bayes + 0.10 * T_gatm +
             0.10 * T_ftl + 0.10 * T_maml)
        return np.clip(T, 0, 1)

    def hlOA_fitness(self, weights, H, T):
        scores = 1 / (1 + np.exp(-np.clip(H @ weights, -500, 500)))
        # High trust -> predicted honest. Correlate with T.
        detection = -np.mean(np.abs(scores - T))  # maximize agreement
        delay_penalty = np.var(scores)
        energy_penalty = np.mean(np.abs(weights))
        return detection - 0.3 * delay_penalty - 0.1 * energy_penalty

    def hlOA_optimize(self, H, T):
        dim = FEATURE_DIM
        pop = np.random.randn(HLOA_POPULATION, dim) * 0.1
        fitness = np.array([self.hlOA_fitness(p, H, T) for p in pop])
        for _ in range(HLOA_ITERATIONS):
            best = pop[np.argmax(fitness)].copy()
            for i in range(HLOA_POPULATION):
                if np.random.rand() < 0.5:
                    pop[i] += 0.1 * (best - pop[i]) + 0.05 * np.random.randn(dim)
                else:
                    pop[i] -= 0.1 * (best - pop[i]) - 0.05 * np.random.randn(dim)
            new_fit = np.array([self.hlOA_fitness(p, H, T) for p in pop])
            improved = new_fit > fitness
            fitness[improved] = new_fit[improved]
        return pop[np.argmax(fitness)]

    def route(self, H):
        T = self.compute_ensemble_trust(H)
        for i in range(self.num_nodes):
            self.bayes.update(i, T[i] > 0.5)
        self.W_edge = self.hlOA_optimize(H, T)
        return T, self.W_edge
