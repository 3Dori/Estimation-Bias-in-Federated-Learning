import logging
from datetime import datetime
from math import sqrt

import numpy as np
import torch.nn.functional as F

from matplotlib import pyplot as plt

from train_fed import FederatedLearningTrainer
import dataset.li_ion_dataset as li_ion_dataset

from model import NeuralNet


def experiment(K=4, sigma=1.0, beta=0.1, batch_size=None, E=1, n_global_rounds=60,
               model_name='MLP',
               use_cuda=False, n_experiments=10):
    result_dict = {'K': K, 'E': E, 'sigma': sigma, 'beta': beta, 'batch_size': batch_size, 'model_name': model_name, 'test_loss': []}
    print(f'Running experiment for K = {K}, sigma = {sigma}, E = {E}, beta = {beta}, model_name = {model_name}')
    for n in range(n_experiments):
        print(f'Experiment {n}')
        train_loaders, test_loader = li_ion_dataset.get_li_ion_dataloader(batch_size=batch_size)
        trainer = FederatedLearningTrainer(train_loaders, test_loader, use_cuda=use_cuda,
                                           K=4, E=E, learning_rate=beta, n_global_rounds=n_global_rounds,
                                           model=NeuralNet, model_parameters={'input_size': 5, 'hidden_size': 4, 'num_classes': 1},
                                           criterion=F.mse_loss,
                                           is_classification=False)
        test_loss = []
        trainer.train(test_loss)
        result_dict['test_loss'].append(test_loss)
        print(f'\tfinal test loss = {test_loss}')
    logging.info(f'Training ended at {datetime.now().strftime("%y%m%d%H%M%S")}')
    return result_dict


def save_result(results):
    import pickle
    from pathlib import Path

    Path('../results_li_ion').mkdir(parents=True, exist_ok=True)
    filename = f'../results_li_ion/results_li_ion_{datetime.now().strftime("%y%m%d%H%M%S")}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    import logging

    import numpy as np

    logging.basicConfig(level=logging.INFO)
    results = []

    sigma_ranges = [0.2, 1.0, 2.0, 5.0, 10.0]
    # Es = [1, 2, 5, 10, 15, 20]
    Es = [10]
    betas = [0.0001]

    for sigma in sigma_ranges:
        sigma = sqrt(sigma)
        for E in Es:
            for beta in betas:
                result_dict = experiment(sigma=sigma, E=E, beta=beta,
                                         use_cuda=True,
                                         n_global_rounds=30, n_experiments=1)
                results.append(result_dict)

    save_result(results)
