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
        return torch.normal(0, sqrt(2)/2, size=(k, 2), device=self.device, dtype=torch.float32).view(torch.complex64)

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
            k = [str(i) for i in range(1, self.k + 1)]
            width = 0.3
            # plt.bar(k, p_h_sqaured, width=width)
            # plt.bar(k + width, eta_tilde, width=width)
            # plt.show()

            # plt.bar(k, p_star, width=width)
            # plt.show()
            plt.figure()
            plt.bar(k, torch.sqrt(p_star) * self.h_norm[sorted_index] / torch.sqrt(eta_star), color='tab:blue')
            plt.xticks(ticks=[], minor=True)
            plt.xlabel('Index of each device, $k$')
            plt.ylabel('$\sqrt{p_k^*} |h_k| / \sqrt{\eta^*}$')
            plt.savefig('../figures/threshold_based_structure_b.pdf', bbox_inches='tight')

            plt.figure()
            plt.bar(k, p_star, color='tab:blue')
            plt.xticks(ticks=[], minor=True)
            plt.xlabel('Index of each device, $k$')
            plt.ylabel('$p_k^*$')
            plt.savefig('../figures/threshold_based_structure_p.pdf', bbox_inches='tight')

        return p_star, eta_star


def get_bias_term(K=10, max_power=1.0, sigma=1.0, n_experiments=250):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    b = 0.
    for n in range(n_experiments):
        pc = PowerControl(K, max_power, sigma, device=device, plot=False)
        b += (pc.h_norm_sqrt_p_star / pc.k_sqrt_eta_star).sum().item()
    return b / n_experiments


def plot_power_control_sigma():
    import numpy as np
    import scienceplots
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    plt.style.use(['science', 'ieee'])

    sigmas = np.arange(0.1, 10.1, 0.1)
    bs = [get_bias_term(sigma=sigma) for sigma in sigmas]
    # plt.figure()

    fig, ax = plt.subplots(figsize=[5,4])
    plt.ylabel('$\\bar{b} = \\frac{1}{K} \\sum_{k=1}^K b_k = \\frac{1}{K} \\sum_{k=1}^K  \\frac{\\sqrt{p_k} |h_k|}{\\sqrt{\\eta}}$')
    plt.xlabel('$\\sigma$')
    ax.plot(sigmas, bs)
    ax.set_xlim(0, 10)
    axins = inset_axes(ax, loc=1, width=2.2, height=1.8)
    x1, x2, y1, y2 = 3.5, 10, 0, 0.02
    data_x = int(len(sigmas) * x1 / 10) - 1
    axins.plot(sigmas[data_x:], bs[data_x:])
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
    ax.grid()
    axins.grid()
    plt.draw()
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
