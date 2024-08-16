import logging
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F

from train_fed import FederatedLearningTrainer
from dataset.MNIST_noniid import get_MNIST_dataloader


def experiment(K=10, is_iid=False, gamma=0.1, sigma=1.0, beta=0.1, batch_size=None, E=1, n_global_rounds=60,
               model_name='LogisticRegression', use_cuda=False, n_experiments=10, random_seed=None):
    result_dict = {'K': K, 'is_iid': is_iid, 'gamma': gamma, 'sigma': sigma, 'beta': beta, 'batch_size': batch_size, 'model_name': model_name, 'E': E, 'test_loss': [], 'accuracy': []}
    print(f'Running experiment for K = {K}, is_iid = {is_iid}, gamma = {gamma}, sigma = {sigma}, beta = {beta}, E = {E}, model_name = {model_name}')
    for n in range(n_experiments):
        print(f'Experiment {n}')
        if random_seed:
            np.random.seed(random_seed)
            torch.random.manual_seed(random_seed)

        train_loaders, test_loader = \
            get_MNIST_dataloader(K=K, is_iid=is_iid, gamma=gamma,
                                batch_size=batch_size, test_batch_size=1000, data_path='../data')
        trainer = FederatedLearningTrainer(train_loaders, test_loader, K=K, E=E, learning_rate=beta, n_global_rounds=n_global_rounds,
                                           model_name=model_name, criterion=F.nll_loss, use_cuda=use_cuda, sigma=sigma)
        test_loss, accuracy = [], []
        trainer.train(test_loss, accuracy)
        result_dict['test_loss'].append(test_loss)
        result_dict['accuracy'].append(accuracy)
        print(f'\tfinal accuracy = {accuracy[-1]}')
    return result_dict


def save_result(results):
    import pickle
    from datetime import datetime
    from pathlib import Path

    Path('../results').mkdir(parents=True, exist_ok=True)
    filename = f'./results_{datetime.now().strftime("%y%m%d%H%M%S")}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    import logging

    import numpy as np

    logging.basicConfig(level=logging.INFO)
    results = []

    # gamma_ranges = [0.1, 0.5, 1.0, 5.0, None]
    # sigma_ranges = [0.2, 1.0, 2.0, 5.0, 10.0]
    # sigma_ranges = [4.0]
    # Es = [10, 20, 50, 100, 1000]
    gamma_ranges = [1.0]
    sigma_ranges = [1.0]
    Es = [10]
    betas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    # Es = [1, 3, 10, 32, 100, 316, 1000, 3162]
    # Es = [10, 20, 30, 40, 50]
    n_global_rounds = 30
    n_experiments = 1

    for gamma in gamma_ranges:
        for sigma in sigma_ranges:
            sigma = sqrt(sigma)
            for E in Es:
                for beta in betas:
                    is_iid = gamma is None
                    gamma = gamma or 10.0
                    result_dict = experiment(K=10, is_iid=is_iid, gamma=gamma, sigma=sigma, beta=beta, E=E,
                                             use_cuda=False,
                                             n_global_rounds=n_global_rounds, n_experiments=n_experiments, random_seed=2024)
                    results.append(result_dict)
    # for sigma in sigma_ranges:
    #     result_dict = experiment(K=10, is_iid=True, gamma=1.0, sigma=sigma, use_cuda=False, n_global_rounds=100, n_experiments=20)
    #     results.append(result_dict)

    save_result(results)
