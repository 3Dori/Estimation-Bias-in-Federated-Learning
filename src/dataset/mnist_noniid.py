import numpy as np


class MNISTNonIIDDataset:
    N_CLASSES = 10

    def __init__(self, original_dataset, K, gamma, random_seed=None):
        self.K = K
        self.gamma = gamma
        self.random_seed = random_seed
        self.n_dataset = len(original_dataset)

        data_idx = np.arange(self.n_dataset)
        targets = original_dataset.targets
        self.train_data_idx_by_label = [data_idx[targets == y] for y in range(self.N_CLASSES)]

    def generate_dataset(self):
        if self.random_seed:
            np.random.seed(self.random_seed)
        distr = np.random.dirichlet([self.gamma] * self.N_CLASSES, self.K)
        size_per_device_per_class = (distr * self.n_dataset / self.K).round().astype(int)

        data_idx = [None] * self.K
        targets = [None] * self.K
        for k in range(self.K):
            data_idx[k] = np.concatenate([np.random.choice(self.train_data_idx_by_label[y], size)
                                          for y, size in enumerate(size_per_device_per_class[k])])
            targets[k] = np.concatenate([[y] * size for y, size in enumerate(size_per_device_per_class[k])])
        return data_idx, targets
    

if __name__ == '__main__':
    from collections import Counter
    from torchvision import datasets
    original_train_dataset = datasets.MNIST('../data', train=True, download=True)
    dataset = MNISTNonIIDDataset(original_train_dataset, K=10, gamma=1.0, random_seed=2024)
    _, targets = dataset.generate_dataset()
    for target_per_device in targets:
        print(Counter(target_per_device))
