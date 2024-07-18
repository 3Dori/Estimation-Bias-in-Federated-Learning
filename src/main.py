import logging

import numpy as np
import torch.nn.functional as F

from matplotlib import pyplot as plt

from power_control import PowerControl
from train_fed import FederatedLearningTrainer
import dataset.MNIST_noniid as MNIST_noniid


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    train_loaders, test_loader = \
        MNIST_noniid.get_MNIST_dataloader(K=10, is_iid=False, gamma=0.1,
                                          batch_size=None, test_batch_size=1000, data_path='../data')
    criterion = F.nll_loss
    trainer = FederatedLearningTrainer(train_loaders, test_loader, K=10, n_global_rounds=10, criterion=criterion)
    test_loss, accuracy = [], []
    trainer.train(test_loss, accuracy)
    print(test_loss)
    print(accuracy)
    plt.plot(np.arange(len(accuracy)), accuracy)
    plt.show()
