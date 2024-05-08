import logging

import torch
from matplotlib import pyplot as plt

from power_control import PowerControl
from train_fed import Trainer
# from train import Trainer as TT


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    trainer = Trainer(K=10, n_global_rounds=10)
    trainer.train()
