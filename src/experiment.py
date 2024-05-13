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


def merge_experiment_results(result_dict1, result_dict2):
    def has_same_key(item1, item2):
        return all(item1[key] == item2[key]
                   for key in ['K', 'is_iid', 'gamma', 'sigma', 'batch_size', 'model_name'])

    # O(n^2)
    for item1 in result_dict1:
        for item2 in result_dict2:
            if has_same_key(item1, item2):
                item1['test_loss'] += item2['test_loss']
                item1['accuracy'] += item2['accuracy']
    return result_dict1


if __name__ == '__main__':
    import pickle
    import logging
    from datetime import datetime

    import numpy as np

    logging.basicConfig(level=logging.INFO)
    results = []

    for gamma in np.arange(0.1, 1.1, 0.1):
        for sigma in np.arange(0.2, 2.2, 0.2):
            result_dict = experiment(K=10, is_iid=False, gamma=gamma, sigma=sigma, use_cuda=True, n_experiments=1)
            results.append(result_dict)

    filename = f'results_{datetime.now().strftime("%y%m%d%H%M%S")}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
