import torch
import torch.nn as nn
from typing import List

class CNN(nn.Module):
    def __init__(self, n_output: int = 1, n_hiddens: List[int] = [64, 32, 16, 8], n_layers: int = 8):
        super(CNN, self).__init__()
        n_penultimate = int(n_hiddens[-1] * (32 / len(n_hiddens) ** 2) ** 2)
        layers = []
        for i in range(len(n_hiddens)):
            n_hidden = n_hiddens[i]
            layers.append(nn.Conv2d(1 if i == 0 else n_hiddens[i - 1], n_hidden, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(n_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.AvgPool2d(2))
        layers.append(nn.Flatten())
        for _ in range(max(0, n_layers - len(n_hiddens))):
            layers.append(nn.Linear(n_penultimate, n_penultimate))
            layers.append(nn.BatchNorm1d(n_penultimate))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_penultimate, n_output))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(
        self, n_input: int = 1, n_output: int = 1, n_hidden: int = 64, n_layers: int = 8
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
    
class LSTM(nn.Module):
    def __init__(
        self, n_input: int = 1, n_output: int = 1, n_hidden: int = 64, n_layers=1
    ):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(n_input, n_hidden, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        out, (h, c) = self.lstm(x)  
        h = h[-1] 
        out = self.fc(h) 
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    cnn = CNN()
    mlp = MLP()
    lstm = LSTM()

    print(f"Number of parameters in CNN: {count_parameters(cnn):,}")
    print(f"Number of parameters in MLP: {count_parameters(mlp):,}")
    print(f"Number of parameters in MLP: {count_parameters(lstm):,}")

if __name__ == '__main__':
    main()