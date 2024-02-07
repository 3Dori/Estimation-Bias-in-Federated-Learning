import numpy as np
from matplotlib import pyplot as plt

from power_control import PowerControl


if __name__ == '__main__':
    K: int = 20
    D: int = 2
    MAX_POW: float = 1.0
    SIGMA: float = 1.0
    h: np.array = np.random.normal(0.0, 1.0, (K, D))
    h_norm: np.array = np.linalg.norm(h, axis=1)
    p_max: np.array = np.full(K, MAX_POW)

    pc = PowerControl(h_norm, p_max, SIGMA)
    p_star, eta_star = pc.compute_optimal_p_and_eta(plot=True)

    # b: np.array = h_norm * np.sqrt(p_star) / np.sqrt(eta_star)
    # plt.bar(np.arange(1, K+1), b)
    # plt.show()
