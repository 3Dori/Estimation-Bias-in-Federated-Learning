from math import isclose

import pickle

import os
import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm


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
    for item1 in result_dict1:
        for item2 in result_dict2:
            if has_same_key(item1, item2):
                results.append(item1)
                item1['test_loss'] += item2['test_loss']
                item1['accuracy'] += item2['accuracy']
                break
    for item1 in result_dict1:
        for item2 in result_dict2:
            if has_same_key(item1, item2):
                break
        else:
            results.append(item1)
    for item2 in result_dict2:
        for item1 in result_dict1:
            if has_same_key(item1, item2):
                break
        else:
            results.append(item2)
    return results


def merge_experiment_results_in_folder(files=None, root='../results/'):
    candidates = []
    if files is not None:
        for file in files:
            if file.endswith('.pkl'):
                with open(file, 'rb') as f:
                    candidates.append(pickle.load(f))
    else:
        for _, _, files in os.walk(root):
            for file in files:
                if file.endswith('.pkl'):
                    with open(os.path.join(root, file), 'rb') as f:
                        candidates.append(pickle.load(f))
    results = []
    for candidate in candidates:
        results = merge_experiment_results(results, candidate)
    return results


def plot_all(results, fixed_param='sigma', variable='gamma', max_epochs=100):
    import scienceplots

    plt.style.use(['science', 'ieee'])

    def get_mean(results, key='accuracy'):
        return np.array([row for row in results[key] if len(row) > 1]).mean(axis=0)

    zs = {round(result[fixed_param], 2) for result in results}
    for z in zs:
        results_by_z = [result for result in results
                        if isclose(result[fixed_param], z, abs_tol=0.001)]
        variables = np.array([result[variable] for result in results_by_z])
        losses = [get_mean(result, key='test_loss') for result in results_by_z]
        # accuracies_for_v = [get_mean(result, key='accuracy') for result in results_by_z]

        plt.figure()
        for v, y in zip(variables, losses):
            if len(y) > max_epochs:
                y = y[:max_epochs]
            x = np.arange(1, len(y) + 1)
            label = 'IID data' if variable == 'gamma' and v == 10.0 else f'${variable}={v}$'
            plt.plot(x, y, label=label)
        plt.legend()
        title = 'IID data' if fixed_param == 'gamma' and z == 10.0 else f'${fixed_param}={z}$'
        plt.title(title)
        plt.savefig(f'../figures/accuracy_{fixed_param}_{z}.pdf', bbox_inches='tight')


def plot_with_param_fixed(results, fixed_param=None, variable='gamma', max_epochs=100):
    import scienceplots

    plt.style.use(['science', 'ieee'])

    def get_mean(results, key='accuracy'):
        return np.array([row for row in results[key] if len(row) > 1]).mean(axis=0)
    
    def is_result_match(result, fixed_param):
        return all(isclose(result[key], value, abs_tol=0.001) for key, value in fixed_param.items())
    
    def gen_fixed_param_str(fixed_param):
        return ','.join(f'{key}={value}' for key, value in fixed_param.items())

    results_by_z = [result for result in results
                    if is_result_match(result, fixed_param)]
    variables = np.array([result[variable] for result in results_by_z])
    losses = [get_mean(result, key='test_loss') for result in results_by_z]
    # accuracies_for_v = [get_mean(result, key='accuracy') for result in results_by_z]

    plt.figure()
    color = cm.rainbow(np.linspace(0, 1, 5))
    for i, (v, y) in enumerate(zip(variables, losses)):
        if variable == 'E' and v == 1000:
            continue
        if len(y) > max_epochs:
            y = y[:max_epochs]
        x = np.arange(1, len(y) + 1)
        label = 'IID data' if variable == 'gamma' and v == 10.0 else f'${variable}={v}$'
        plt.plot(x, y, label=label, c=color[i])
    plt.legend()
    title = 'IID data' if fixed_param.get('gamma', 0) == 10.0 else f'${gen_fixed_param_str(fixed_param)}$'
    plt.title(title)
    plt.savefig(f'../figures/accuracy_{gen_fixed_param_str(fixed_param)}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    with open('../results/results_E_gt_1_0712.pkl', 'rb') as f:
        results = pickle.load(f)
    plot_with_param_fixed(results, fixed_param={'gamma': 1.0, 'sigma': 1.0}, variable='E')
    # plot_with_param_fixed(results, fixed_param='gamma', variable='sigma')