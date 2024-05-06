from typing import Tuple

import torch
from matplotlib import pyplot as plt


class PowerControl:
    """
    Reproduce the power control optimization from
    'Optimized Power Control for Over-the-Air Computation in Fading Channels'
    by Xiaowen Gao et al.
    """
    def __init__(self, k: int, p_max: float, sigma: float, device) -> None:
        self.device = device

        self.k = k
        self.h = self._generate_h(k)
        self.h_norm = torch.linagl.vector_norm(self.h, axis=1)
        self.p_max = torch.full(k, p_max, device=device)
        self.sigma = sigma

        self.p_star, self.eta_star = self.compute_optimal_p_and_eta()
        self.b = self.h_norm * torch.sqrt(self.p_star) / torch.sqrt(self.eta_star)
        self.k_sqrt_eta_star = self.k * torch.sqrt(self.eta_star)

    def _generate_h(self, k):
        return torch.normal(0, torch.sqrt(2)/2, size=(k, 2), device=self.device).view(torch.complex128)
    
    def _generate_z(self, shape):
        return torch.normal(0, self.sigma, size=shape, device=self.device)
    
    def receive(self, w):
        assert w.shape[0] == self.k
        x = torch.sum(self.b[:,] * w, axis=0)
        z = self._generate_z(shape=x.shape)
        return (x + z) / self.k_sqrt_eta_star

    def compute_optimal_p_and_eta(self, plot: bool = False):
        sorted_p_h_sqaured = self.p_max * (self.h_norm ** 2)
        sorted_index = torch.argsort(sorted_p_h_sqaured)
        rank = torch.argsort(sorted_index)

        p_star, eta_star = self._compute_optimal_p_and_eta_with_sorted_p_h_sqaured(sorted_index, plot)
        return p_star[rank], eta_star

    def _compute_optimal_p_and_eta_with_sorted_p_h_sqaured(self, sorted_index, plot: bool):
        p_max = self.p_max[sorted_index]
        h_squared = (self.h_norm ** 2)[sorted_index]
        p_h_sqaured = p_max * h_squared

        sum_1_to_i_pi_hi_squared = torch.cumsum(p_h_sqaured)
        sum_1_to_i_sqrt_pi_hi = torch.cumsum(torch.sqrt(p_h_sqaured))
        eta_tilde = ((self.sigma ** 2 + sum_1_to_i_pi_hi_squared) / sum_1_to_i_sqrt_pi_hi) ** 2
        k_star = torch.argmax(eta_tilde < p_h_sqaured)
        eta_star = eta_tilde[k_star]
        p_star = torch.where(eta_tilde >= p_h_sqaured, p_max, eta_star / h_squared)

        if plot:
            k = torch.arange(1, self.k+1)
            width = 0.3
            plt.bar(k, p_h_sqaured, width=width)
            plt.bar(k + width, eta_tilde, width=width)
            plt.show()

            plt.bar(k, p_star, width=width)
            plt.show()

        return p_star, eta_star
