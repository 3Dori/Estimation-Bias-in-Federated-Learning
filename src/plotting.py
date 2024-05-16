from math import isclose

import pickle

import os
import os.path

import numpy as np
from matplotlib import pyplot as plt


def merge_experiment_results(result_dict1, result_dict2):
    def has_same_key(item1, item2):
        def eq(value1, value2):
            if isinstance(value1, float):
                return isclose(value1, value2, abs_tol=0.01)
            else:
                return value1 == value2
        return all(eq(item1[key], item2[key])
                   for key in ['K', 'is_iid', 'gamma', 'sigma', 'batch_size', 'model_name'])

    results = []
    keys = []
    for item1 in result_dict1:
        for item2 in result_dict2:
            if has_same_key(item1, item2):
                results.append(item1)
                item1['test_loss'] += item2['test_loss']
                item1['accuracy'] += item2['accuracy']
                break
    for item2 in result_dict2:
        for item1 in result_dict1:
            if has_same_key(item1, item2):
                break
        else:
            results.append(item2)
    return results


def merge_experiment_results_in_folder(root='../results/'):
    candidates = []
    for _, _, files in os.walk(root):
        for file in files:
            if file.endswith('.pkl'):
                with open(os.path.join(root, file), 'rb') as f:
                    candidates.append(pickle.load(f))
    results = []
    for candidate in candidates:
        results = merge_experiment_results(results, candidate)
    return results


def plot_with_param_fixed(results, fixed_param='sigma', variable='gamma'):
    def get_mean(results, key='accuracy'):
        return np.array([row for row in results[key] if len(row) > 1]).mean(axis=0)

    zs = {round(result[fixed_param], 2) for result in results}
    for z in zs:
        results_by_z = [result for result in results
                        if isclose(result[fixed_param], z, abs_tol=0.001)]
        variables = np.array([result[variable] for result in results_by_z])
        losses = [get_mean(result, key='test_loss') for result in results_by_z]
        accuracies_for_v = [get_mean(result, key='accuracy') for result in results_by_z]
        # for v, accuracies in zip(variables, accuracies_for_v):
        #     x = np.arange(1, len(accuracies) + 1)
        #     plt.plot(x, accuracies, label=f'gamma={v}')

        for v, loss in zip(variables, losses):
            x = np.arange(1, len(loss) + 1)
            plt.plot(x, loss, label=f'gamma={v}')
        plt.legend()
        plt.title(f'\${fixed_param}\$ = {z}')
        plt.show()


if __name__ == '__main__':
    with open('../merged_results/results_15.pkl', 'rb') as f:
        results = pickle.load(f)
    plot_with_param_fixed(results, fixed_param='sigma', variable='gamma')