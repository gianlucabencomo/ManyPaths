import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class ModuloDataset(Dataset):
    def __init__(self, n_samples: int = 10000, m: int = 10, range_max: int = 100):
        self.n_samples = n_samples
        self.m = m
        self.range_max = range_max
        self.X = torch.randint(0, self.range_max, size=(self.n_samples,)).float().unsqueeze(1)
        self.y = (self.X % self.m).float()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MetaModuloDataset(Dataset):
    def __init__(self, n_tasks: int = 10, n_samples_per_task: int = 100, range_max: int = 100):
        self.n_tasks = n_tasks
        self.n_samples_per_task = n_samples_per_task
        self.range_max = range_max

        # Pre-generate data for each task
        self.tasks = []
        for m in range(1, n_tasks + 1):  # Each task corresponds to a different modulo value
            X = torch.randint(0, range_max, size=(n_samples_per_task,)).float().unsqueeze(1)
            y = (X % m).float()
            self.tasks.append((X, y, m))  # Store X, y, and the task-specific modulo value

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, idx):
        return self.tasks[idx]

    def sample_task(self):
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, n_input: int = 1, n_output: int = 1, n_hidden: int = 64, n_layers: int = 2):
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

dataset = ModuloDataset(n_samples=128, m=20, range_max=100)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

# Initialize the model, loss function, and optimizer
model = MLP(n_layers=8)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

# Training loop
epochs = 1000
losses = []
for epoch in range(epochs):
    model.train()
    batch_loss = []
    X = dataset.X
    y = dataset.y
    y_pred = model(X)
    loss = criterion(y_pred, y)
    writer.add_scalar("Loss/train", loss, epoch)
        
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    losses.append(loss.item())
    #losses.append(np.mean(batch_loss))
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Plot loss over epochs
plt.plot(range(epochs), losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

X = dataset.X
y = dataset.y
sorted_indices = torch.argsort(X, dim=0)
x_data_sorted = X[sorted_indices].squeeze(2)
y_data_sorted = y[sorted_indices].squeeze(2)

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(x_data_sorted)

# Plot the results
plt.plot(x_data_sorted.numpy(), y_data_sorted.numpy(), label="True Function")
plt.plot(x_data_sorted.numpy(), y_pred.numpy(), label="MLP Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
writer.close()