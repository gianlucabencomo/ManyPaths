import typer
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import learn2learn as l2l


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
        indices = [self.token2idx[token] for token in sequence]
        return torch.tensor(indices, dtype=torch.long)

    def collate_fn(self, batch):
        batch = pad_sequence(
            batch, batch_first=True, padding_value=self.token2idx["<PAD>"]
        )
        return batch[:, :-1], batch[:, 1:]


def samples_to_sequences(samples):
    sequences = []
    for sample in samples:
        seq = ["<SOS>"] + ["A"] * sample + ["<EOS>"]
        sequences.append(seq)
    return sequences


class MetaRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size):
        super(MetaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded, hidden)
        output = self.fc(rnn_out)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


def main(
    seed: int = 0,
    scale: float = 1.0,
    n_samples: int = 100,
    batch_size: int = 32,
    hidden_size: int = 32,
    meta_lr: float = 0.001,
    fast_lr: float = 0.01,
    n_inner_steps: int = 1,
    n_meta_epochs: int = 10,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    samples, lam = make_poisson(n=n_samples, scale=scale)
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

    # Define the meta model
    model = MetaRNN(
        vocab_size=len(vocab), hidden_size=hidden_size, output_size=len(vocab)
    )

    # Wrap model with MAML
    maml = l2l.algorithms.MAML(model, lr=fast_lr)

    # Define meta-optimizer
    meta_optimizer = optim.Adam(maml.parameters(), lr=meta_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    # Meta-training loop
    for epoch in range(n_meta_epochs):
        total_meta_loss = 0
        for episode, (support_set, targets) in enumerate(train_loader):
            # Clone the model for inner-loop adaptation
            learner = maml.clone()
            hidden = learner.init_hidden(support_set.size(0))

            # Inner loop: Adaptation on the support set
            for _ in range(n_inner_steps):
                learner.zero_grad()
                outputs, hidden = learner(support_set, hidden)
                outputs = outputs.view(-1, outputs.shape[-1])
                targets = targets.view(-1)
                inner_loss = criterion(outputs, targets)
                inner_loss.backward()
                learner.adapt(inner_loss)

            # Outer loop: Compute meta-loss on the query set
            query_outputs, hidden = learner(support_set, hidden)
            query_outputs = query_outputs.view(-1, query_outputs.shape[-1])
            query_loss = criterion(query_outputs, targets.view(-1))
            total_meta_loss += query_loss.item()

            # Meta-update
            meta_optimizer.zero_grad()
            query_loss.backward()
            meta_optimizer.step()

        avg_meta_loss = total_meta_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{n_meta_epochs}], Meta Loss: {avg_meta_loss:.4f}")

    # Meta-testing (evaluation) on new episodes with unseen Gaussians
    evaluate_meta_model(maml, val_loader, criterion)


def evaluate_meta_model(maml, dataloader, criterion, n_inner_steps=1):
    maml.eval()
    total_loss = 0

    with torch.no_grad():
        for support_set, targets in dataloader:
            learner = maml.clone()
            hidden = learner.init_hidden(support_set.size(0))

            # Adaptation on the support set
            for _ in range(n_inner_steps):
                outputs, hidden = learner(support_set, hidden)
                outputs = outputs.view(-1, outputs.shape[-1])
                targets = targets.view(-1)
                inner_loss = criterion(outputs, targets)
                learner.adapt(inner_loss)

            # Evaluate on the adapted model
            outputs, hidden = learner(support_set, hidden)
            outputs = outputs.view(-1, outputs.shape[-1])
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Meta-test Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    typer.run(main)
