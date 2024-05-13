from math import sqrt

import torch
import torch.linalg
from matplotlib import pyplot as plt


class PowerControl:
    """
    Reproduce the power control optimization from
    'Optimized Power Control for Over-the-Air Computation in Fading Channels'
    by Xiaowen Gao et al.
    """
    def __init__(self, k: int, p_max: float, sigma: float, device, plot: bool = False) -> None:
        self.device = device

        self.k = k
        self.h = self._generate_h(k)
        self.h_norm = torch.linalg.vector_norm(self.h, axis=1)
        self.p_max = torch.full((k,), p_max, device=device)
        self.sigma = sigma

        self.p_star, self.eta_star = self.compute_optimal_p_and_eta(plot=plot)
        self.h_norm_sqrt_p_star = self.h_norm * torch.sqrt(self.p_star)
        self.k_sqrt_eta_star = self.k * torch.sqrt(self.eta_star)

    def _generate_h(self, k):
        return torch.normal(0, sqrt(2)/2, size=(k, 2), device=self.device, dtype=torch.float32).view(torch.complex32)
    
    def _generate_z(self, shape):
        return torch.normal(0, self.sigma, size=shape, device=self.device)
    
    def receive(self, delta_w):
        assert delta_w.shape[0] == self.k
        mu = delta_w.mean(dim=1)
        sigma = delta_w.std(dim=1)
        delta_w = (delta_w - mu[:,None]) / sigma[:,None]
        x = torch.matmul(self.h_norm_sqrt_p_star, delta_w)
        z = self._generate_z(shape=x.shape)
        y = (x + z) / self.k_sqrt_eta_star
        return (y * sigma.mean()) + mu.mean()

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

        sum_1_to_i_pi_hi_squared = torch.cumsum(p_h_sqaured, dim=0)
        sum_1_to_i_sqrt_pi_hi = torch.cumsum(torch.sqrt(p_h_sqaured), dim=0)
        eta_tilde = ((self.sigma ** 2 + sum_1_to_i_pi_hi_squared) / sum_1_to_i_sqrt_pi_hi) ** 2
        k_star = torch.argmax((eta_tilde < p_h_sqaured) * 1.0)
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

            plt.bar(k, torch.sqrt(p_star) * self.h_norm[sorted_index] / (self.k * torch.sqrt(eta_star)))
            plt.show()

        return p_star, eta_star
    

def get_bias_term(K=10, max_power=1.0, sigma=1.0, n_experiments=50):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    b = 0.
    for n in range(n_experiments):
        pc = PowerControl(K, max_power, sigma, device=device, plot=False)
        b += (pc.h_norm_sqrt_p_star / pc.k_sqrt_eta_star).sum().item()
    return b / n_experiments


if __name__ == '__main__':
    import numpy as np
    import scienceplots

    plt.style.use(['science', 'ieee'])

    torch.manual_seed(2024)
    sigmas = np.arange(0.1, 2.1, 0.1)
    bs = [get_bias_term(sigma=sigma) for sigma in sigmas]
    plt.figure()
    plt.plot(sigmas, bs)
    plt.grid()
    plt.ylim(0.0, 1.0)
    plt.ylabel('Bias Term B')
    plt.xlabel('$\\sigma$')
    plt.savefig('../figures/power_control_sigma_vs_bias.pdf', bbox_inches="tight")

    # K: int = 10
    # N = 100
    # MAX_POW: float = 1.0
    # SIGMA: float = 1.0
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # pc = PowerControl(K, MAX_POW, SIGMA, device=device, plot=True)
    # x = torch.full((K, N), 1.0)
    # w = pc.receive(x)
    # plt.bar(torch.arange(N), w)
    # plt.show()