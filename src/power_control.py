from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


class PowerControl:
    """
    Reproduce the power control optimization from
    'Optimized Power Control for Over-the-Air Computation in Fading Channels'
    by Xiaowen Gao et al.
    """
    def __init__(self, h_norm: np.array, p_max: np.array, sigma: np.float32) -> None:
        self.k: int = len(h_norm)
        self.h_norm = h_norm
        self.p_max = p_max
        self.sigma = sigma

    def compute_optimal_p_and_eta(self, plot: bool = True) -> Tuple[np.array, np.float32, np.array]:
        sorted_p_h_sqaured: np.array = self.p_max * (self.h_norm ** 2)
        sorted_index: np.array = np.argsort(sorted_p_h_sqaured)
        rank: np.array = np.argsort(sorted_index)

        p_star, eta_star = self._compute_optimal_p_and_eta_with_sorted_p_h_sqaured(sorted_index, plot)
        return p_star[rank], eta_star

    def _compute_optimal_p_and_eta_with_sorted_p_h_sqaured(self, sorted_index: np.array, plot: bool):
        p_max: np.array = self.p_max[sorted_index]
        h_squared: np.array = (self.h_norm ** 2)[sorted_index]
        p_h_sqaured = p_max * h_squared

        sum_1_to_i_pi_hi_squared: np.array = np.cumsum(p_h_sqaured)
        sum_1_to_i_sqrt_pi_hi: np.array = np.cumsum(np.sqrt(p_h_sqaured))
        eta: np.array = ((self.sigma ** 2 + sum_1_to_i_pi_hi_squared) / sum_1_to_i_sqrt_pi_hi) ** 2
        k_star: np.int64 = np.argmax(eta < p_h_sqaured)
        eta_star: np.float32 = eta[k_star]
        p_star: np.array = np.where(eta >= p_h_sqaured, p_max, eta_star / h_squared)

        if plot:
            k: np.array = np.arange(1, self.k+1)
            width = 0.3
            plt.bar(k, p_h_sqaured, width=width)
            plt.bar(k + width, eta, width=width)
            plt.show()

            plt.bar(k, p_star, width=width)
            plt.show()

        return p_star, eta_star
