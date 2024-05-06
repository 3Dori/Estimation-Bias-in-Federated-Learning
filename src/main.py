import logging

import torch
from matplotlib import pyplot as plt

from power_control import PowerControl
from train_fed import Trainer


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    K: int = 20
    MAX_POW: float = 1.0
    SIGMA: float = 1.0

    # pc = PowerControl(K, MAX_POW, SIGMA, device=torch.device('cpu'), plot=False)
    # print(pc.receive(torch.normal(0, 1, size=(K, 10))))

    trainer = Trainer()
    trainer.train()