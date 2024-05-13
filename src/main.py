import logging

import numpy as np

from matplotlib import pyplot as plt
import scienceplots

from power_control import PowerControl
from train_fed import FederatedLearningTrainer
import dataset.MNIST_noniid as MNIST_noniid
# from train import Trainer as TT


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    train_loaders, test_loader = \
        MNIST_noniid.get_MNIST_dataloader(K=10, is_iid=False, gamma=0.1,
                                          batch_size=None, test_batch_size=1000, data_path='../data')
    trainer = FederatedLearningTrainer(train_loaders, test_loader, K=10, n_global_rounds=10)
    test_loss, accuracy = [], []
    trainer.train(test_loss, accuracy)
    print(test_loss)
    print(accuracy)
    plt.plot(np.arange(len(accuracy)), accuracy)
    plt.title('Example')
    plt.show()
