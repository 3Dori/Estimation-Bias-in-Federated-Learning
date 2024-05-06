import logging

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms

from model import Net
from power_control import PowerControl


class FederatedLearningDataset(torch.utils.data.Dataset):
    """
    Generate K datasets from the original dataset for federated learning.
    """
    def __init__(self, original_dataset: torch.utils.data.Dataset, K: int, k: int):
        self.original_dataset = original_dataset
        self.K = K
        if k < 0 or k >= K:
            raise ValueError('k must be in the range [0, K - 1]')
        self.k = k

    def __len__(self):
        return len(self.original_dataset) // self.K

    def __getitem__(self, idx):
        transformed_idx = len(self) * self.k + idx
        return self.original_dataset[transformed_idx]


class Trainer:
    MNIST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def __init__(self, K=10, E=1, n_global_rounds=30,
                 batch_size=None, test_batch_size=1000, data_path='../data',
                 learning_rate=0.01):
        self.K = K
        self.E = E
        self.n_global_rounds = n_global_rounds
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.train_loaders = self._get_MNIST_dataloader(K, batch_size, data_path)
        self.models = [Net().to(self.device) for k in range(K)]
        self.optimizers = [torch.optim.SGD(model.parameters(), lr=learning_rate) for model in self.models]

        self.global_model = Net().to(self.device)
        self.global_optimizer = torch.optim.SGD(self.global_model.parameters())

        test_dataset = datasets.MNIST(data_path, train=False, transform=Trainer.MNIST_TRANSFORM)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)

        self._num_parameters = sum(param.numel() for param in self.global_model.parameters())

        self.power_control = PowerControl(K, p_max=1.0, sigma=1.0)
    
    def train(self):
        for global_round in range(self.n_global_rounds):
            self._broadcast_weights()
            self._local_train()
            self._global_update()
            self._test()
            logging.info(f'Global round: {global_round}/{self.n_global_rounds} '
                         f'({global_round / self.n_global_rounds:.0f}%)')

    def _local_train(self):
        for k in range(self.K):
            for epoch in range(self.E):
                self._train(k)

    def _combine_local_weights(self):
        w = np.zeros((self.K, self._num_parameters), dtype=np.float32)
        with torch.no_grad():
            for k in range(self.K):
                model = self.models[k]
                start_idx = 0
                for param in model.parameters():
                    w[k,start_idx:start_idx + param.numel()] = torch.flatten(param)
                    start_idx += param.numel()
                assert start_idx == self._num_parameters
        return w

    def _global_update(self):
        w = self._combine_local_weights()
        global_w = self.power_control.receive(w)
        start_idx = 0
        with torch.no_grad():
            for param in self.global_model.parameters():
                param.copy_(global_w[start_idx:start_idx + param.numel()])
                start_idx += param.numel()
        assert start_idx == self._num_parameters

    def _broadcast_weights(self):
        with torch.no_grad():
            for k in range(self.K):
                model = self.models[k]
                for local_param, global_param in zip(model.parameters(), self.global_model.parameters()):
                    local_param.copy_(global_param)

    def _train(self, k):
        dataloader, model, optimizer = self.train_loaders[k], self.models[k], self.optimizers[k]
        model.train()
        for _, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
        logging.info(f'Local train for device {k}')
    
    def _test(self):
        self.global_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
            
        test_loss /= len(self.test_loader.dataset)
        logging.info(f'Test set: Average loss: {test_loss:.4f}, '
                     f'Accuracy: {correct}/{len(self.test_loader.dataset)} '
                     f'({100. * correct / len(self.test_loader.dataset):.1f}%)')

    @staticmethod
    def _get_MNIST_dataloader(K, batch_size, data_path):
        original_train_dataset = datasets.MNIST(data_path, train=True, download=True,
                                                transform=Trainer.MNIST_TRANSFORM)
        train_datasets = [FederatedLearningDataset(original_train_dataset, K, k) for k in range(K)]
        train_loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size or len(dataset))
                         for dataset in train_datasets]
        return train_loaders
