import logging

import torch
from matplotlib import pyplot as plt

from power_control import PowerControl
from train_fed import Trainer
from train import Trainer as TT


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # trainer = Trainer(K=10)
    trainer = TT(epochs=1)
    trainer.train()
    print(list(trainer.model.parameters())[0])