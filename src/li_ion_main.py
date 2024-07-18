import logging

import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataset import li_ion_dataset
from train_fed import FederatedLearningTrainer
from model import NeuralNet


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    train_loaders, test_loader = li_ion_dataset.get_li_ion_dataloader(batch_size=120)
    model = NeuralNet
    model_parameters = {'input_size': 5, 'hidden_size': 4, 'num_classes': 1}
    trainer = FederatedLearningTrainer(train_loaders, test_loader,
                                       K=4, E=1, n_global_rounds=10,
                                       model=model, criterion=F.mse_loss, model_parameters=model_parameters,
                                       is_classification=False)
    test_loss = []
    trainer.train(test_loss)
    plt.plot(np.arange(len(test_loss)), test_loss)
    plt.title('Example')
    plt.show()