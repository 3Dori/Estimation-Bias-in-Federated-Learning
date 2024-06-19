import logging

import numpy as np

from matplotlib import pyplot as plt

from train_fed import FederatedLearningTrainer
from dataset.MNIST_noniid import get_MNIST_dataloader


def experiment(K=10, is_iid=False, gamma=0.1, sigma=1.0, batch_size=None, n_global_rounds=60,
               model_name='LogisticRegression', use_cuda=False, n_experiments=10):
    result_dict = {'K': K, 'is_iid': is_iid, 'gamma': gamma, 'sigma': sigma, 'batch_size': batch_size, 'model_name': model_name, 'test_loss': [], 'accuracy': []}
    print(f'Running experiment for K = {K}, is_iid = {is_iid}, gamma = {gamma}, sigma = {sigma}, model_name = {model_name}')
    for n in range(n_experiments):
        print(f'Experiment {n}')
        train_loaders, test_loader = \
            get_MNIST_dataloader(K=K, is_iid=is_iid, gamma=gamma,
                                batch_size=batch_size, test_batch_size=1000, data_path='../data')
        trainer = FederatedLearningTrainer(train_loaders, test_loader, K=K, n_global_rounds=n_global_rounds,
                                           model_name=model_name, use_cuda=use_cuda, sigma=sigma)
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

    # gamma_ranges = [0.1, 0.5, 1.0, 5.0, 10.0]
    sigma_ranges = [0.2, 1.0, 2.0, 10.0]

    # for gamma in gamma_ranges:
    #     for sigma in sigma_ranges:
    #         result_dict = experiment(K=10, is_iid=False, gamma=gamma, sigma=sigma, use_cuda=False, n_global_rounds=100, n_experiments=20)
    #         results.append(result_dict)
    for sigma in sigma_ranges:
        result_dict = experiment(K=10, is_iid=True, gamma=1.0, sigma=sigma, use_cuda=False, n_global_rounds=100, n_experiments=20)
        results.append(result_dict)

    save_result(results)
