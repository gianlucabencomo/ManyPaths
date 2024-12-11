import torch
import torch.nn as nn
import torch.optim as optim

from datasets import MetaBitStringModuloDataset
from models import LSTM

# Hyperparameters
lr = 0.001
epochs = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the dataset
dataset = MetaBitStringModuloDataset(n_tasks=20, n_samples_per_task=100, range_max=100)

# Sample one task (e.g., the first one)
X_bits_s, X_s, y_s, X_bits_q, X_q, y_q, m = dataset.tasks[15]

X_bits_s = X_bits_s.to(device)  # (N, bit_width, 1)
y_s = y_s.to(device)            # (N, 1)
X_bits_q = X_bits_q.to(device)  # (range_max, bit_width, 1)
y_q = y_q.to(device)            # (range_max, 1)

model = LSTM(n_hidden=64, n_layers=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
model.train()
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(X_bits_s)
    loss = criterion(y_pred, y_s)
    # Backprop
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate on the query set
model.eval()
with torch.no_grad():
    y_q_pred = model(X_bits_q)
    eval_loss = criterion(y_q_pred, y_q)
    print(f"Query Set Loss: {eval_loss.item():.4f}")

    # Optional: print a few predictions vs actual
    for i in range(50):
        print(f"Input: {X_q[i].item()} Mod {m} => True: {y_q[i].item()}, Pred: {y_q_pred[i].item():.4f}")

from visualize import plot_loss
import matplotlib.pyplot as plt

plot_loss(losses)
plt.show()