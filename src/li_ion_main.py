import logging

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataset import li_ion_dataset
from train_fed import FederatedLearningTrainer
from model import NeuralNet


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    load_voltage = True

    train_loaders, test_loader = li_ion_dataset.get_li_ion_dataloader(batch_size=120, test_batch_size=None, load_voltage=load_voltage)

    trainer = FederatedLearningTrainer(train_loaders, test_loader,
                                       K=4, E=5, n_global_rounds=20,
                                       model=NeuralNet, model_parameters={'input_size': 4, 'hidden_size': 4, 'num_classes': 1},
                                       criterion=F.mse_loss,
                                       sigma=0.2,
                                       is_classification=False)
    test_loss = []
    trainer.train(test_loss)

    model = trainer.global_model

    with torch.no_grad():
        import scienceplots
        plt.style.use(['science', 'ieee'])

        test_input, test_target = test_loader.dataset.x, test_loader.dataset.y
        test_output = model(test_input)
        time = np.arange(0, 6000, 6000 / len(test_target))
        y_name = 'Voltage' if load_voltage else 'SOC'
        plt.plot(time, test_output, 'b-', label=f'Estimated {y_name}', linewidth=0.5)
        plt.plot(time, test_target, 'r-', label=f'Measured {y_name}', linewidth=0.5)
        plt.xlabel('Time/s')
        if load_voltage:
            plt.ylabel('Voltage/V')
        else:
            plt.ylabel('SOC/\%')
        plt.legend()
        if load_voltage:
            plt.savefig('../figures/FL-Voltage.pdf', bbox_inches='tight')
        else:
            plt.savefig('../figures/FL-SOC.pdf', bbox_inches='tight')
