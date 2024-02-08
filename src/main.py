import numpy as np
from matplotlib import pyplot as plt

from power_control import PowerControl


if __name__ == '__main__':
    K: int = 20
    MAX_POW: float = 1.0
    SIGMA: float = 1.0

    pc = PowerControl(K, MAX_POW, SIGMA)
    print(pc.receive(np.random.normal(0, 1, (K, 10, 20))))

    # b: np.array = h_norm * np.sqrt(p_star) / np.sqrt(eta_star)
    # plt.bar(np.arange(1, K+1), b)
    # plt.show()
