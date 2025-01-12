import math
import copy
import typer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import List, Optional

from constants import CNN_PARAMS, MLP_PARAMS, LSTM_PARAMS, TRANSFORMER_PARAMS


class CNN(nn.Module):
    def __init__(
        self,
        n_input_channels: int = 1,
        n_output: int = 1,
        n_hiddens: List[int] = [64, 32, 16],
        n_layers: int = 5,
    ):
        super().__init__()
        n_penultimate = int(n_hiddens[-1] * (32 / 2 ** len(n_hiddens)) ** 2)
        layers = []
        for i, n_hidden in enumerate(n_hiddens):
            layers.append(
                nn.Conv2d(
                    n_input_channels if i == 0 else n_hiddens[i - 1], n_hidden, 3, 1, 1
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
        n_input_channels: int = 1,
    ):
        super().__init__()
        layers = []
        if n_input < 64:
            layers.extend(
                [
                    nn.Linear(n_input, 32 * 32 * n_input_channels),
                    nn.BatchNorm1d(32 * 32 * n_input_channels),
                    nn.ReLU(),
                    nn.Linear(32 * 32 * n_input_channels, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                ]
            )
        else:
            layers.extend(
                [
                    nn.Linear(n_input, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                ]
            )
        layers.append(nn.ReLU())
        for _ in range(n_layers - 2):
            layers.extend(
                [
                    nn.Linear(n_hidden, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.ReLU(),
                ]
            )
        layers.append(nn.Linear(n_hidden, n_output))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LSTMCell(nn.Module):
    def __init__(
        self, n_input: int = 16, n_hidden: int = 64, use_layer_norm: bool = True
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.gates = nn.Linear(n_hidden + n_input, 4 * n_hidden)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.gates_norm = nn.LayerNorm(4 * n_hidden)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        z = torch.cat((h, x), dim=1)
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
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.input_proj = nn.Linear(n_input, n_hidden)
        self.lstm = nn.ModuleList(
            [LSTMCell(n_hidden, n_hidden) for i in range(n_layers)]
        )
        self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = [
            torch.zeros(batch_size, self.n_hidden, device=x.device)
            for _ in range(self.n_layers)
        ]
        c = [
            torch.zeros(batch_size, self.n_hidden, device=x.device)
            for _ in range(self.n_layers)
        ]
        for t in range(seq_len):
            input_t = self.input_proj(x[:, t, :])
            for layer in range(self.n_layers):
                h[layer], c[layer] = self.lstm[layer](input_t, h[layer], c[layer])
                input_t = h[layer]
        return self.fc(h[-1])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].requires_grad_(False)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 128,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=0.0, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src2 = self.norm1(src)
        attn_output, _ = self.self_attn(
            src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + attn_output
        src2 = self.norm2(src)
        src = src + self.linear2(torch.relu(self.linear1(src2)))
        return src


class Transformer(nn.Module):
    def __init__(
        self,
        n_input: int = 16,
        n_output: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 2 * 64,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.decoder = nn.Linear(d_model, n_output)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.encoder:
            x = layer(x)
        return self.decoder(x[:, -1, :])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_parameter_ranges(
    cnn_range,
    mlp_range,
    lstm_range,
    transformer_range,
    cnn_values,
    mlp_values,
    lstm_values,
    transformer_values,
    index=0,
):
    # Prepare data
    architectures = ["CNN", "MLP", "LSTM", "Transformer"]
    min_values = [cnn_range[0], mlp_range[0], lstm_range[0], transformer_range[0]]
    max_values = [cnn_range[1], mlp_range[1], lstm_range[1], transformer_range[1]]
    all_values = [cnn_values, mlp_values, lstm_values, transformer_values]

    # Create a number line plot
    plt.figure(figsize=(12, 3))
    for i, arch in enumerate(architectures):
        # Plot the range line
        plt.hlines(y=i, xmin=min_values[i], xmax=max_values[i], color="blue", lw=3)

        # Plot individual values as black dots
        for value in all_values[i]:
            plt.scatter(value, i, color="black", zorder=5, linewidths=3)

        # Plot the min and max points
        plt.scatter(min_values[i], i, color="red", zorder=5, linewidths=3)  # Min point
        plt.scatter(
            max_values[i], i, color="green", zorder=5, linewidths=3
        )  # Max point

    def format_func(value, tick_number):
        if value == 0:
            return "0"
        else:
            return f"{int(value / 1000)}K"

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

    # Add labels and title
    plt.yticks(range(len(architectures)), architectures, fontsize=16)
    plt.xlabel("Number of Parameters", fontsize=16)
    plt.xticks(fontsize=16)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    # Add annotations for min and max values
    for i, arch in enumerate(architectures):
        plt.text(
            min_values[i],
            i + 0.14,
            f"{min_values[i]:,}",
            color="red",
            ha="center",
            fontsize=12,
        )
        plt.text(
            max_values[i],
            i - 0.32,
            f"{max_values[i]:,}",
            color="green",
            ha="center",
            fontsize=12,
        )

    y_min = -0.35  # Add buffer below
    y_max = 3.35  # Add buffer above
    plt.ylim(y_min, y_max)

    plt.tight_layout(pad=0)
    plt.savefig(
        f"parameter_ranges{index}.pdf", format="pdf", dpi=300, bbox_inches="tight"
    )
    plt.show()


def main(index: int = 3):
    C = [1, 3, 1, 1, 3]
    counts_mlp = [1, 4, 8, 32 * 32 * 1, 32 * 32 * 3]
    counts_seq = [1, 1, 1, 16, 48]

    cnn_max, mlp_max, lstm_max, transformer_max = 0, 0, 0, 0
    cnn_min, mlp_min, lstm_min, transformer_min = np.inf, np.inf, np.inf, np.inf
    cnn_values, mlp_values, lstm_values, transformer_values = [], [], [], []
    for i in range(len(CNN_PARAMS)):
        cnn = CNN(C[index], 1, CNN_PARAMS[i][0], CNN_PARAMS[i][1])
        mlp = MLP(
            n_input=counts_mlp[index],
            n_hidden=MLP_PARAMS[i][0],
            n_layers=MLP_PARAMS[i][1],
            n_input_channels=C[index],
        )
        lstm = LSTM(
            n_input=counts_seq[index],
            n_hidden=LSTM_PARAMS[i][0],
            n_layers=LSTM_PARAMS[i][1],
        )
        transformer = Transformer(
            n_input=counts_seq[index],
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
    print(
        f"Number of parameters in Transformer: min = {transformer_min}, max = {transformer_max}"
    )

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
        index=index,
    )


if __name__ == "__main__":
    typer.run(main)
