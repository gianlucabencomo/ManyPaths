import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np

from generate_numbers import generate_number_grid

class CNNModuloDataset(Dataset):
    def __init__(self, n_samples: int = 10000, m: int = 10, range_max: int = 100):
        self.n_samples = n_samples
        self.m = m
        self.range_max = range_max
        self.X = torch.randint(0, self.range_max, size=(self.n_samples,)).float().unsqueeze(1)
        self.y = (self.X % self.m).float()

        self.X_images = torch.zeros((n_samples, 1, 32, 32))

        for i, number in enumerate(self.X[:,0].int().numpy()):
            grid_image = generate_number_grid(number).reshape(1, 32, 32)
            self.X_images[i] = torch.tensor(grid_image)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN(nn.Module):
    def __init__(self, n_output: int = 1):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 32x32 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 16x16 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, n_output),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

dataset = CNNModuloDataset(n_samples=128, m=20, range_max=100)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

# Initialize the model, loss function, and optimizer
model = CNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

# Training loop
epochs = 1000
losses = []
for epoch in range(epochs):
    model.train()
    batch_loss = []
    X = dataset.X_images
    y = dataset.y
    y_pred = model(X)
    loss = criterion(y_pred, y)
        
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
