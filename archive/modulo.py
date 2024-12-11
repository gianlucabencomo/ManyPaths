import typer

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split


class MLP(nn.Module):
    def __init__(
        self, n_input: int = 1, n_output: int = 1, n_hidden: int = 64, n_layers: int = 2
    ):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(n_input, n_hidden))
        layers.append(nn.BatchNorm1d(n_hidden))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.BatchNorm1d(n_hidden))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(n_hidden, n_output))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ModuloDataset(Dataset):
    def __init__(self, n_samples: int = 10000, m: int = 3):
        self.X = torch.randint(0, 100, size=(n_samples,)).float().unsqueeze(1)
        self.y = torch.tensor([self.modulo(x.item(), m) for x in self.X]).unsqueeze(1)

    @staticmethod
    def modulo(X, m):
        return X % m

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    seed: int = 0,
    n_samples: int = 10000,
    modulo: int = 10,
    n_hidden: int = 128,
    n_layers: int = 3,
    batch_size: int = 10000,
    epochs: int = 10000,
    learning_rate: int = 1e-2,
):
    # set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set device
    device = get_device()

    # set up data + dataloader
    dataset = ModuloDataset(n_samples=n_samples, m=modulo)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # init model, criterion, optimizer
    model = MLP(n_hidden=n_hidden, n_layers=n_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )


if __name__ == "__main__":
    typer.run(main)
