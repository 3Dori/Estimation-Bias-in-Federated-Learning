from math import isclose

import pickle

import os
import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    

def save_pkl(d, path):
    with open(path, 'wb') as f:
        pickle.dump(d, f)


def merge_experiment_results_from_path(path1, path2):
    result_dict1 = load_pkl(path1)
    result_dict2 = load_pkl(path2)
    return merge_experiment_results(result_dict1, result_dict2)


def merge_experiment_results(result_dict1, result_dict2):
    def has_same_key(item1, item2):
        def eq(value1, value2):
            if isinstance(value1, float):
                return isclose(value1, value2, abs_tol=0.0001)
            else:
                return value1 == value2
        return all(eq(item1[key], item2[key])
                   for key in ['K', 'E', 'beta', 'is_iid', 'gamma', 'sigma', 'batch_size', 'model_name'])

    results = []
    for item1 in result_dict1:
        for item2 in result_dict2:
            if has_same_key(item1, item2):
                results.append(item1)
                item1['test_loss'] += item2['test_loss']
                item1['accuracy'] += item2['accuracy']
                break
    for item1 in result_dict1:
        if not any(has_same_key(item1, item2) for item2 in result_dict2):
            results.append(item1)
    for item2 in result_dict2:
        if not any(has_same_key(item1, item2) for item1 in result_dict1):
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


def plot_all(results, fixed_param='sigma', variable='gamma', max_epochs=100, inset_axis=True):
    import scienceplots

    plt.style.use(['science', 'ieee'])

    def get_mean(results, key='accuracy'):
        return np.array([row for row in results[key] if len(row) > 1]).mean(axis=0)

    zs = {round(result[fixed_param], 2) for result in results}
    for z in zs:
        results_by_z = [result for result in results
                        if isclose(result[fixed_param], z, abs_tol=0.01)]
        variables = np.array([result[variable] for result in results_by_z])
        losses = [get_mean(result, key='test_loss') for result in results_by_z]
        # accuracies_for_v = [get_mean(result, key='accuracy') for result in results_by_z]

        color = cm.rainbow(np.linspace(0, 1, len(variables)))
        fig, ax = plt.subplots(figsize=[5,4])
        if inset_axes:
            axins = ax.inset_axes([0.21, 0.6, 0.5, 0.33])
            x1, x2, y1, y2 = 25, 30, 0.29, 0.37
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
        for i, (v, y) in enumerate(zip(variables, losses)):
            y = np.hstack((np.array([2.5988]), y))
            if len(y) > max_epochs:
                y = y[:max_epochs]
            x = np.arange(0, len(y) + 0)
            if variable == 'sigma':
                label = f'$\\sigma^2={v**2:.1f}$'
            elif variable == 'gamma' and v == 10.0:
                label = 'IID data'
            else:
                label = f'$\\{variable}={v}$'
            ax.plot(x, y, '-', label=label, c=color[i], linewidth=0.6)
            if inset_axes:
                axins.plot(x, y, '-', label=label, c=color[i], linewidth=0.6)
        if fixed_param == 'sigma':
            title = f'$\\sigma^2={z**2:.1f}$'
        elif fixed_param == 'gamma' and z == 10.0:
            title = 'IID data'
        else:
            title = f'$\\{fixed_param}={z}$'
        ax.legend()
        ax.set_xlabel('Communication round')
        ax.set_ylabel('Test loss')
        ax.set_title(title)
        plt.savefig(f'../figures/test_loss_{fixed_param}_{z}.pdf', bbox_inches='tight')


def plot_with_param_fixed(results, fixed_param=None, variable='gamma', max_epochs=150, inset_axes=True):
    import scienceplots

    plt.style.use(['science', 'ieee'])

    def get_mean(results, key='accuracy'):
        return np.array([row for row in results[key] if len(row) > 1]).mean(axis=0)
    
    def is_result_match(result, fixed_param):
        return all(isclose(result[key], value, abs_tol=0.001) for key, value in fixed_param.items())
    
    def gen_fixed_param_str(fixed_param):
        return ','.join(f'\\{key}={value}' for key, value in fixed_param.items())

    results_by_z = [result for result in results
                    if is_result_match(result, fixed_param)]
    variables = np.array([result[variable] for result in results_by_z])
    losses = [get_mean(result, key='test_loss') for result in results_by_z]
    # accuracies_for_v = [get_mean(result, key='accuracy') for result in results_by_z]

    fig, ax = plt.subplots(figsize=[5,4])
    if inset_axes:
        axins = ax.inset_axes([0.2, 0.6, 0.5, 0.33])
        x1, x2, y1, y2 = 140, 150, 0.28, 0.283
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
    color = cm.rainbow(np.linspace(0, 1, len(variables)))
    for i, (v, y) in enumerate(zip(variables, losses)):
        if variable == 'E' and v == 1000:
            continue
        if len(y) > max_epochs:
            y = y[:max_epochs]
        x = np.arange(1, len(y) + 1)
        if variable == 'sigma':
            label = f'$\\sigma^2={v**2:.1f}$'
        elif variable == 'gamma' and v == 10.0:
            label = 'IID data'
        else:
            label = f'$\\{variable}={v}$'
        ax.plot(x, y, '-', label=label, c=color[i], linewidth=0.6)
        if inset_axes:
            axins.plot(x, y, '-', label=label, c=color[i], linewidth=0.6)
    ax.legend()
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Test loss')
    # title = 'IID data' if fixed_param.get('gamma', 0) == 10.0 else f'${gen_fixed_param_str(fixed_param)}$'
    # ax.set_title(title)
    plt.savefig(f'../figures/li_loss_{gen_fixed_param_str(fixed_param)}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    # with open('../results/results_E_10_gamma_sigma.pkl', 'rb') as f:
    #     results = pickle.load(f)
    # plot_all(results, fixed_param='gamma', variable='sigma')
    # plot_all(results, fixed_param='sigma', variable='gamma')

    # Experiment with E
    # with open('../results/results_E_10_to_50.pkl', 'rb') as f:
    #     results = pickle.load(f)
    # plot_with_param_fixed(results, fixed_param={'gamma': 1.0, 'sigma': 1.0}, variable='E')
    # plot_with_param_fixed(results, fixed_param='gamma', variable='sigma')

    # Experiment with beta
    # with open('../results/results_beta.pkl', 'rb') as f:
    #     results = pickle.load(f)
    # plot_with_param_fixed(results, fixed_param={'gamma': 1.0, 'sigma': 1.0, 'E': 10}, variable='beta')

    # with open('../results/results_beta_0.1_0.14_0.15.pkl', 'rb') as f:
    #     results = pickle.load(f)
    # plot_with_param_fixed(results, fixed_param={'gamma': 1.0, 'sigma': 1.0, 'E': 10}, variable='beta')

    # Experiment with E - Li-ion
    # with open('../results_li_ion/results_different_sigmas.pkl', 'rb') as f:
    #     results = pickle.load(f)
    # plot_with_param_fixed(results, fixed_param={'E': 10}, variable='sigma', inset_axes=False)
    # plot_with_param_fixed(results, fixed_param='gamma', variable='sigma')

    # Experiment with beta - Li-ion
    with open('../results_li_ion/results_li_ion_different_betas.pkl', 'rb') as f:
        results = pickle.load(f)
    plot_with_param_fixed(results, fixed_param={'E': 10}, variable='beta', inset_axes=False)