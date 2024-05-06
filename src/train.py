import logging

import torch
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms

from model import Net


class Trainer:
    def __init__(self, batch_size=64, test_batch_size=1000, data_path='../data',
                 learning_rate=0.01, epochs=30, logging_interval=50):
        self.epochs = epochs
        self.logging_interval = logging_interval
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        train_kwargs = {'batch_size': batch_size}
        test_kwargs = {'batch_size': test_batch_size}

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(data_path, train=True, download=True,
                                       transform=transform)
        test_dataset = datasets.MNIST(data_path, train=False, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        self.model = Net().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def train(self):
        for epoch in range(self.epochs):
            self._train(epoch)
            self._test()

    def save(self):
        torch.save(self.model.state_dict(), "mnist_cnn.pt")

    def _train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.logging_interval == 0:
                logging.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)}'
                             f'({100. * batch_idx / len(self.train_loader):.1f}%)]\tLoss: {loss.item():.6f}')

    def _test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        logging.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)}'
                     f'({100. * correct / len(self.test_loader.dataset):.1f}%)')
