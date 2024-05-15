import logging

import numpy as np

import torch
from torchvision import datasets, transforms


class FederatedLearningDataset(torch.utils.data.Dataset):
    """
    Generate datasets from the original dataset for federated learning, given the indices on each device.
    """
    def __init__(self, original_dataset: torch.utils.data.Dataset, data_idx):
        super().__init__()
        self.original_dataset = original_dataset
        self.data_idx = data_idx

    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        return self.original_dataset[self.data_idx[idx]]


class MNISTNonIIDDataset:
    N_CLASSES = 10

    def __init__(self, original_dataset, K, gamma):
        self.K = K
        self.gamma = gamma
        self.n_dataset = len(original_dataset)

        data_idx = np.arange(self.n_dataset)
        targets = original_dataset.targets
        self.train_data_idx_by_label = [data_idx[targets == y] for y in range(self.N_CLASSES)]

    def generate_dataset(self):
        for train_data_idx in self.train_data_idx_by_label:
            np.random.shuffle(train_data_idx)

        samples_per_class = np.array([len(data) for data in self.train_data_idx_by_label])
        distr = np.random.dirichlet([self.gamma] * self.K, self.N_CLASSES)
        splits = distr.cumsum(axis=1) * samples_per_class[:,np.newaxis]
        splits = np.hstack([np.zeros((self.N_CLASSES, 1)), splits]).astype(int)

        data_idx = [None] * self.K
        targets = [None] * self.K
        for k in range(self.K):
            data_idx[k] = np.concatenate([train_data_idx[split[k]:split[k+1]]
                                          for train_data_idx, split in zip(self.train_data_idx_by_label, splits)])
            targets[k] = np.concatenate([[y] * (split[k+1] - split[k])
                                         for y, split in enumerate(splits)])
        return data_idx, targets
    

def get_MNIST_dataloader(K, batch_size=None, test_batch_size=1000, data_path='../data',
                         is_iid=True, gamma=0.1):
    MNIST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    original_train_dataset = datasets.MNIST(data_path, train=True, download=True,
                                            transform=MNIST_TRANSFORM)
    if is_iid:
        size_per_device = len(original_train_dataset) // K
        fed_datasets = [
            FederatedLearningDataset(original_train_dataset,
                                        data_idx=np.arange(k*size_per_device, (k+1)*size_per_device))
            for k in range(K)]
    else:
        data_idx, _ = MNISTNonIIDDataset(original_train_dataset, K, gamma).generate_dataset()
        fed_datasets = [FederatedLearningDataset(original_train_dataset, idx) for idx in data_idx]

    train_loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size or len(dataset), pin_memory=True)
                     for dataset in fed_datasets]
    
    test_dataset = datasets.MNIST(data_path, train=False, transform=MNIST_TRANSFORM)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, pin_memory=True)
    logging.info(f'Dataset prepared with is_iid = {is_iid}, gamma = {gamma}')

    return train_loaders, test_loader


if __name__ == '__main__':
    from collections import Counter

    original_train_dataset = datasets.MNIST('../data', train=True, download=True)
    dataset = MNISTNonIIDDataset(original_train_dataset, K=10, gamma=0.1)
    _, targets = dataset.generate_dataset()
    for target_per_device in targets:
        print(Counter(target_per_device))
