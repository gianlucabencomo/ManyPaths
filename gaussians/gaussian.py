import typer
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def make_poisson(n: int = 100, scale: float = 1.0):
    lam = np.random.gamma((1,), scale=scale)
    samples = np.random.poisson(lam, size=(n,))
    return samples, lam


class SequenceDataset(Dataset):
    def __init__(self, sequences, vocab):
        self.sequences = sequences
        self.vocab = vocab
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Convert tokens to indices
        indices = [self.token2idx[token] for token in sequence]
        return torch.tensor(indices, dtype=torch.long)

    def collate_fn(self, batch):
        # Pad sequences to the same length
        batch = pad_sequence(
            batch, batch_first=True, padding_value=self.token2idx["<PAD>"]
        )
        return (
            batch[:, :-1],
            batch[:, 1:],
        )  # Inputs and targets for next-token prediction


def samples_to_sequences(samples):
    sequences = []
    for sample in samples:
        seq = ["<SOS>"] + ["A"] * sample + ["<EOS>"]
        sequences.append(seq)
    return sequences


# class SimpleTransformer(nn.Module):
#     def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2):
#         super(SimpleTransformer, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
#         self.fc_out = nn.Linear(d_model, vocab_size)
#         self.vocab_size = vocab_size

#     def forward(self, src):
#         embedded = self.embedding(src)
#         transformer_out = self.transformer(embedded)
#         output = self.fc_out(transformer_out)
#         return output


class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.i2h = nn.Linear(hidden_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input)  # (batch_size, seq_len, hidden_size)
        output_seq = []
        for i in range(embedded.size(1)):  # Loop through sequence length
            embedded_t = embedded[:, i, :]  # (batch_size, hidden_size)
            hidden = F.tanh(self.i2h(embedded_t) + self.h2h(hidden))
            output = self.h2o(hidden)
            output_seq.append(output)

        # Stack all outputs to form the final output sequence
        output_seq = torch.stack(
            output_seq, dim=1
        )  # (batch_size, seq_len, output_size)
        return output_seq, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


def train_model(model, dataloader, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )  # Ignore padding token in loss computation

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()

            # Initialize hidden state for each batch
            hidden = model.init_hidden(inputs.size(0))

            outputs, hidden = model(
                inputs, hidden
            )  # Outputs: (batch_size, seq_len, vocab_size)
            outputs = outputs.reshape(
                -1, outputs.shape[-1]
            )  # Flatten for loss computation
            targets = targets.reshape(-1)  # Flatten targets

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader)}")


def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(
        ignore_index=0, reduction="sum"
    )  # Use sum for perplexity calculation

    with torch.no_grad():
        for inputs, targets in dataloader:
            hidden = model.init_hidden(inputs.size(0))
            outputs, hidden = model(inputs, hidden)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = outputs.argmax(dim=1)
            mask = targets != 0  # Ignore padding tokens
            total_correct += (predictions == targets).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    accuracy = total_correct / total_tokens
    print(
        f"Evaluation - Perplexity: {perplexity.item():.4f}, Accuracy: {accuracy * 100:.2f}%"
    )
    return perplexity.item(), accuracy


def main(
    seed: int = 0,
    scale: float = 1.0,
    n_samples: int = 100,
    batch_size: int = 32,
    hidden_size: int = 32,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    samples, lam = make_poisson(n=n_samples, scale=scale)
    print(samples)

    sequences = samples_to_sequences(samples)
    vocab = ["<PAD>", "<SOS>", "<EOS>", "A"]

    # Split into training and validation sets
    split_idx = int(0.8 * len(sequences))
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]

    # Create Dataset and DataLoader for training and validation
    train_dataset = SequenceDataset(train_sequences, vocab)
    val_dataset = SequenceDataset(val_sequences, vocab)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )

    # Initialize the model
    model = RNN(vocab_size=len(vocab), hidden_size=hidden_size, output_size=len(vocab))

    # Evaluate the model
    evaluate_model(model, val_loader)

    # Train the model
    train_model(model, train_loader, epochs=10)

    # Evaluate the model
    evaluate_model(model, val_loader)


if __name__ == "__main__":
    typer.run(main)
