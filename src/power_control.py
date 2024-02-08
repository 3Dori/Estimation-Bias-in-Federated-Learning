from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


class PowerControl:
    """
    Reproduce the power control optimization from
    'Optimized Power Control for Over-the-Air Computation in Fading Channels'
    by Xiaowen Gao et al.
    """
    def __init__(self, k: int, p_max: float, sigma: float) -> None:
        self.k = k
        self.h = self._generate_h(k)
        self.h_norm = np.linalg.norm(self.h, axis=1)
        self.p_max = np.full(k, p_max)
        self.sigma = sigma

        self.p_star, self.eta_star = self.compute_optimal_p_and_eta()
        self.b = self.h_norm * np.sqrt(self.p_star) / np.sqrt(self.eta_star)
        self.k_sqrt_eta_star = self.k * np.sqrt(self.eta_star)

    @staticmethod
    def _generate_h(k) -> np.array:
        return np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(k, 2)).view(np.complex128)
    
    def _generate_z(self, shape) -> np.array:
        return np.random.normal(0, self.sigma, shape)
    
    def receive(self, w) -> np.array:
        assert w.shape[0] == self.k
        x: np.array = np.sum(self.b[:,np.newaxis,np.newaxis] * w, axis=0)
        z: np.array = self._generate_z(shape=x.shape)
        return (x + z) / self.k_sqrt_eta_star

    def compute_optimal_p_and_eta(self, plot: bool = False) -> Tuple[np.array, np.float32, np.array]:
        sorted_p_h_sqaured: np.array = self.p_max * (self.h_norm ** 2)
        sorted_index: np.array = np.argsort(sorted_p_h_sqaured)
        rank: np.array = np.argsort(sorted_index)

        p_star, eta_star = self._compute_optimal_p_and_eta_with_sorted_p_h_sqaured(sorted_index, plot)
        return p_star[rank], eta_star

    def _compute_optimal_p_and_eta_with_sorted_p_h_sqaured(self, sorted_index: np.array, plot: bool) -> Tuple[np.array, np.float32, np.array]:
        p_max: np.array = self.p_max[sorted_index]
        h_squared: np.array = (self.h_norm ** 2)[sorted_index]
        p_h_sqaured = p_max * h_squared

        sum_1_to_i_pi_hi_squared: np.array = np.cumsum(p_h_sqaured)
        sum_1_to_i_sqrt_pi_hi: np.array = np.cumsum(np.sqrt(p_h_sqaured))
        eta_tilde: np.array = ((self.sigma ** 2 + sum_1_to_i_pi_hi_squared) / sum_1_to_i_sqrt_pi_hi) ** 2
        k_star: np.int64 = np.argmax(eta_tilde < p_h_sqaured)
        eta_star: np.float32 = eta_tilde[k_star]
        p_star: np.array = np.where(eta_tilde >= p_h_sqaured, p_max, eta_star / h_squared)

        if plot:
            k: np.array = np.arange(1, self.k+1)
            width = 0.3
            plt.bar(k, p_h_sqaured, width=width)
            plt.bar(k + width, eta_tilde, width=width)
            plt.show()

            plt.bar(k, p_star, width=width)
            plt.show()

        return p_star, eta_star
