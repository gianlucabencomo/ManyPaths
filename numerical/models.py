import torch
import torch.nn as nn
from typing import List
import math
from matplotlib.ticker import FuncFormatter

import numpy as np
import matplotlib.pyplot as plt


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

class LSTMCell(nn.Module):
    def __init__(self, n_input: int = 16, n_hidden: int = 64, use_layer_norm: bool = True):
        super(LSTMCell, self).__init__()
        self.n_hidden = n_hidden
        self.gates = nn.Linear(n_hidden + n_input, 4 * n_hidden)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.gates_norm = nn.LayerNorm(4 * n_hidden)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        z = torch.cat((h, x), dim=1)  # (batch_size, n_hidden + n_input)
 
        gates = self.gates(z)
        if self.use_layer_norm:
            gates = self.gates_norm(gates)
        f, i, c_hat, o = gates.chunk(4, dim=1)
        f, i, o = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o)
        c_hat = torch.tanh(c_hat)

        c = f * c + i * c_hat
        h = o * torch.tanh(c)
        return h, c

class LSTM(nn.Module):
    def __init__(
        self, n_input: int = 16, n_output: int = 1, n_hidden: int = 64, n_layers=2
    ):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = nn.ModuleList([
            LSTMCell(n_input=n_input if i == 0 else n_hidden, n_hidden=n_hidden) for i in range(n_layers)
        ])
        self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        h = [torch.zeros(batch_size, self.n_hidden, device=x.device) for _ in range(self.n_layers)]
        c = [torch.zeros(batch_size, self.n_hidden, device=x.device) for _ in range(self.n_layers)]

        for t in range(seq_len):
            input_t = x[:, t, :]  # Extract input for time step t
            for layer in range(self.n_layers):
                h[layer], c[layer] = self.lstm[layer](input_t, h[layer], c[layer])
                input_t = h[layer]  # Pass hidden state to the next layer

        # Use the last hidden state from the final layer
        out = self.fc(h[-1])  # (batch_size, n_output)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
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
        dim_feedforward: int = 2 * 64,
        dropout: float = 0.0,
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
            norm_first=True,
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


def plot_parameter_ranges(cnn_range, mlp_range, lstm_range, transformer_range, cnn_values, mlp_values, lstm_values, transformer_values):
    # Prepare data
    architectures = ['CNN', 'MLP', 'LSTM', 'Transformer']
    min_values = [cnn_range[0], mlp_range[0], lstm_range[0], transformer_range[0]]
    max_values = [cnn_range[1], mlp_range[1], lstm_range[1], transformer_range[1]]
    all_values = [cnn_values, mlp_values, lstm_values, transformer_values]

    # Create a number line plot
    plt.figure(figsize=(12, 5))
    for i, arch in enumerate(architectures):
        # Plot the range line
        plt.hlines(y=i, xmin=min_values[i], xmax=max_values[i], color='blue', lw=3)
        
        # Plot individual values as black dots
        for value in all_values[i]:
            plt.scatter(value, i, color='black', zorder=5, linewidths=3)

        # Plot the min and max points
        plt.scatter(min_values[i], i, color='red', zorder=5, linewidths=3)  # Min point
        plt.scatter(max_values[i], i, color='green', zorder=5, linewidths=3)  # Max point

    def format_func(value, tick_number):
        if value == 0:
            return '0'
        else:
            return f'{int(value / 1000)}K'

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

    # Add labels and title
    plt.yticks(range(len(architectures)), architectures, fontsize=16)
    plt.xlabel('Number of Parameters', fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Add annotations for min and max values
    for i, arch in enumerate(architectures):
        plt.text(min_values[i], i + 0.1, f"{min_values[i]:,}", color='red', ha='center', fontsize=12)
        plt.text(max_values[i], i - 0.2, f"{max_values[i]:,}", color='green', ha='center', fontsize=12)

    y_min = -0.35 # Add buffer below
    y_max = 3.35  # Add buffer above
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig("parameter_ranges.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    from hyperparameter import CNN_PARAMS, MLP_PARAMS, LSTM_PARAMS, TRANSFORMER_PARAMS

    cnn_max, mlp_max, lstm_max, transformer_max = 0, 0, 0, 0
    cnn_min, mlp_min, lstm_min, transformer_min = np.inf, np.inf, np.inf, np.inf
    cnn_values, mlp_values, lstm_values, transformer_values = [], [], [], []

    for i in range(len(CNN_PARAMS)):
        cnn = CNN(1, CNN_PARAMS[i][0], CNN_PARAMS[i][1])
        mlp = MLP(n_input=32 * 32, n_hidden=MLP_PARAMS[i][0], n_layers=MLP_PARAMS[i][1])
        lstm = LSTM(n_input=16, n_hidden=LSTM_PARAMS[i][0], n_layers=LSTM_PARAMS[i][1])
        transformer = Transformer(
            n_input=16,
            d_model=TRANSFORMER_PARAMS[i][0],
            dim_feedforward=TRANSFORMER_PARAMS[i][0] * 2,
            num_layers=TRANSFORMER_PARAMS[i][1],
        )
        # Count parameters
        n_cnn = count_parameters(cnn)
        cnn_values.append(n_cnn)
        n_mlp = count_parameters(mlp)
        mlp_values.append(n_mlp)
        n_lstm = count_parameters(lstm)
        lstm_values.append(n_lstm)
        n_transformer = count_parameters(transformer)
        transformer_values.append(n_transformer)

        # Update min and max
        if n_mlp > mlp_max:
            mlp_max = n_mlp
        if n_mlp < mlp_min:
            mlp_min = n_mlp
        if n_cnn > cnn_max:
            cnn_max = n_cnn
        if n_cnn < cnn_min:
            cnn_min = n_cnn
        if n_lstm > lstm_max:
            lstm_max = n_lstm
        if n_lstm < lstm_min:
            lstm_min = n_lstm
        if n_transformer > transformer_max:
            transformer_max = n_transformer
        if n_transformer < transformer_min:
            transformer_min = n_transformer

    print(cnn_values[-1])
    print(f"Number of parameters in MLP: min = {mlp_min}, max = {mlp_max}")
    print(f"Number of parameters in CNN: min = {cnn_min}, max = {cnn_max}")
    print(f"Number of parameters in LSTM: min = {lstm_min}, max = {lstm_max}")
    print(f"Number of parameters in Transformer: min = {transformer_min}, max = {transformer_max}")

    # Plot parameter ranges
    plot_parameter_ranges(
        cnn_range=(cnn_min, cnn_max),
        mlp_range=(mlp_min, mlp_max),
        lstm_range=(lstm_min, lstm_max),
        transformer_range=(transformer_min, transformer_max),
        cnn_values=cnn_values,
        mlp_values=mlp_values,
        lstm_values=lstm_values,
        transformer_values=transformer_values,
    )

if __name__ == "__main__":
    main()
