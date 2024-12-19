import torch
import torch.nn as nn
from typing import List
import math


class CNN(nn.Module):
    def __init__(
        self,
        n_output: int = 1,
        n_hiddens: List[int] = [64, 32, 16],
        n_layers: int = 5,
    ):
        super(CNN, self).__init__()
        n_penultimate = int(n_hiddens[-1] * (32 / 2 ** len(n_hiddens)) ** 2)
        layers = []
        for i in range(len(n_hiddens)):
            n_hidden = n_hiddens[i]
            layers.append(
                nn.Conv2d(
                    1 if i == 0 else n_hiddens[i - 1],
                    n_hidden,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
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
        self,
        n_input: int = 32 * 32,
        n_output: int = 1,
        n_hidden: int = 64,
        n_layers: int = 8,
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
        self, n_input: int = 16, n_output: int = 1, n_hidden: int = 64, n_layers=2
    ):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(n_input, n_hidden, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_input: int = 16,
        n_output: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ):
        super(Transformer, self).__init__()

        # Project input to d_model
        self.input_proj = nn.Linear(n_input, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Final linear layer
        self.fc = nn.Linear(d_model, n_output)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        out = self.fc(out[:, -1, :])
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    cnn = CNN(1, [16, 8], 2)
    mlp = MLP(n_input=32 * 32, n_hidden=128, n_layers=10)
    lstm = LSTM(n_hidden=64, n_layers=5)
    transformer = Transformer(n_input=16)

    print(f"Number of parameters in MLP: {count_parameters(mlp):,}")
    print(f"Number of parameters in CNN: {count_parameters(cnn):,}")
    print(f"Number of parameters in LSTM: {count_parameters(lstm):,}")
    print(f"Number of parameters in Transformer: {count_parameters(transformer):,}")


if __name__ == "__main__":
    main()
