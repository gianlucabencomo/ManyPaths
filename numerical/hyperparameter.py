import typer

import numpy as np

import torch
import torch.nn as nn
import learn2learn as l2l
import matplotlib.pyplot as plt


from datasets import (
    MetaNumberModuloDataset,
    MetaImageModuloDataset,
    MetaBitStringModuloDataset,
)
from models import MLP, CNN, LSTM, Transformer
from utils import set_random_seeds
from visualize import plot_loss, plot_meta_test_results

MLP_PARAMS = [
    (16, 2),
    (32, 2),
    (64, 2),
    (128, 2),
    (16, 4),
    (32, 4),
    (64, 4),
    (128, 4),
    (16, 6),
    (32, 6),
    (64, 6),
    (128, 6),
    (16, 8),
    (32, 8),
    (64, 8),
    (128, 8),
    (16, 10),
    (32, 10),
    (64, 10),
    (128, 10),
]
CNN_PARAMS = [
    ([16, 8], 2),
    ([32, 16], 2),
    ([64, 32], 2),
    ([128, 64], 2),
    ([16, 8], 4),
    ([32, 16, 8], 4),
    ([64, 32, 16], 4),
    ([128, 64, 32], 4),
    ([16, 8], 6),
    ([32, 16, 8], 6),
    ([64, 32, 16], 6),
    ([128, 64, 32, 16], 6),
    ([16, 8], 8),
    ([32, 16, 8], 8),
    ([64, 32, 16, 8], 8),
    ([128, 64, 32, 16], 8),
    ([16, 8], 10),
    ([32, 16, 8], 10),
    ([64, 32, 16, 8], 10),
    ([128, 64, 32, 16], 10),
]
LSTM_PARAMS = [
    (16, 1),
    (32, 1),
    (64, 1),
    (128, 1),
    (16, 2),
    (32, 2),
    (64, 2),
    (128, 2),
    (16, 3),
    (32, 3),
    (64, 3),
    (128, 3),
    (16, 4),
    (32, 4),
    (64, 4),
    (128, 4),
    (16, 5),
    (32, 5),
    (64, 5),
    (128, 5),
]
TRANSFORMER_PARAMS = [
    (16, 1),
    (32, 1),
    (64, 1),
    (128, 1),
    (16, 2),
    (32, 2),
    (64, 2),
    (128, 2),
    (16, 3),
    (32, 3),
    (64, 3),
    (128, 3),
    (16, 4),
    (32, 4),
    (64, 4),
    (128, 4),
    (16, 5),
    (32, 5),
    (64, 5),
    (128, 5),
]
models = ["cnn", "mlp", "transformer", "lstm",]
skips = [1]
indices = np.arange(20)


def evaluate(
    meta, dataset, criterion, device, adaptation_steps=[0, 1], return_results=False
):
    meta.train()
    meta_loss, results = 0.0, []
    for task in dataset.tasks:
        X_s, X_num_s, y_s, X_q, X_num_q, y_q, m = task
        X_s, y_s, X_q, y_q = (
            X_s.to(device),
            y_s.to(device),
            X_q.to(device),
            y_q.to(device),
        )
        preds, losses = [], []
        for steps in adaptation_steps:
            learner = meta.clone()
            # Adaptation on the support set
            for _ in range(steps):
                support_pred = learner(X_s)
                support_loss = criterion(support_pred, y_s)
                learner.adapt(support_loss)

            # Evaluate on the query set (post-adaptation)
            with torch.no_grad():
                pred = learner(X_q)
                preds.append(pred)
                losses.append(criterion(pred, y_q).item())

        results.append(
            {
                "m": m,
                "X_s": X_num_s,
                "y_s": y_s,
                "X_q": X_num_q,
                "y_q": y_q,
                "predictions": preds,
                "losses": losses,
            }
        )
        meta_loss += losses[1]  # adaptation step = 1

    meta_loss /= len(dataset.tasks)
    if return_results:
        return meta_loss, results
    else:
        return meta_loss


def meta_train(
    meta,
    train_dataset,
    test_dataset,
    criterion,
    optimizer,
    device,
    epochs: int = 10000,
    tasks_per_meta_batch: int = 4,
    adaptation_steps: int = 1,
):
    meta.train()
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        optimizer.zero_grad()
        meta_loss = 0.0

        for _ in range(tasks_per_meta_batch):
            # Sample a task
            X_s, _, y_s, X_q, _, y_q, _ = train_dataset.sample_task()
            X_s, y_s, X_q, y_q = (
                X_s.to(device),
                y_s.to(device),
                X_q.to(device),
                y_q.to(device),
            )

            # Adaptation on the support set
            learner = meta.clone()
            for _ in range(adaptation_steps):
                support_pred = learner(X_s)
                support_loss = criterion(support_pred, y_s)
                learner.adapt(support_loss)

            # Evaluate on the query set
            query_pred = learner(X_q)
            query_loss = criterion(query_pred, y_q)
            meta_loss += query_loss

        # Meta-update
        meta_loss /= tasks_per_meta_batch
        meta_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            train_loss = evaluate(
                meta, train_dataset, criterion, device, [0, adaptation_steps]
            )
            test_loss = evaluate(
                meta, test_dataset, criterion, device, [0, adaptation_steps]
            )
            train_losses.append(train_loss)
            test_losses.append(test_loss)

    return train_losses, test_losses


def init_dataset_and_model(model, skip, index):
    n_input = 1024 if model == "mlp" else 16

    # init datasets
    train_dataset = MetaImageModuloDataset(
        n_tasks=20,
        n_samples_per_task=20,
        range_max=100,
        skip=skip,
        train=True,
        model=model,
    )
    test_dataset = MetaImageModuloDataset(
        n_tasks=20,
        n_samples_per_task=20,
        range_max=100,
        skip=skip,
        train=False,
        model=model,
    )
    test_train_dataset = MetaImageModuloDataset(
        n_tasks=20,
        n_samples_per_task=20,
        range_max=100,
        skip=skip,
        train=True,
        model=model,
    )

    if model == "mlp":
        n_hidden, n_layers = MLP_PARAMS[index]
        model = MLP(n_input=n_input, n_hidden=n_hidden)
    elif model == "cnn":
        n_hiddens, n_layers = CNN_PARAMS[index]
        model = CNN(n_hiddens=n_hiddens, n_layers=n_layers)
    elif model == "lstm":
        n_hidden, n_layers = LSTM_PARAMS[index]
        model = LSTM(n_input=n_input, n_hidden=n_hidden, n_layers=n_layers)
    else:
        n_hidden, n_layers = TRANSFORMER_PARAMS[index]
        model = Transformer(
            n_input=n_input,
            d_model=n_hidden,
            dim_feedforward=n_hidden,
            num_layers=n_layers,
        )

    return train_dataset, test_dataset, test_train_dataset, model


def main(
    seed: int = 10e3,
    epochs: int = 1000,
    tasks_per_meta_batch: int = 4,  # static
    adaptation_steps: int = 1,  # train [1] test [0, 1]
    inner_lr: float = 1e-3,
    outer_lr: float = 1e-3,
):
    # init dataset and model
    results = []
    for i, (m, skip, index) in enumerate(
        (m, skip, index) for m in models for index in indices for skip in skips
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else ("cpu" if m in ['mlp', 'lstm'] else 'mps'))
        set_random_seeds(seed)
        train_dataset, test_dataset, test_train_dataset, model = init_dataset_and_model(
            m, skip, index
        )

        # init meta-learner, loss, and meta-optimizer
        model = model.to(device)
        meta = l2l.algorithms.MetaSGD(model, lr=inner_lr, first_order=False).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(meta.parameters(), lr=outer_lr)

        # meta-training
        train_losses, test_losses = meta_train(
            meta,
            train_dataset,
            test_dataset,
            criterion,
            optimizer,
            device,
            epochs,
            tasks_per_meta_batch,
            adaptation_steps,
        )
        print(
            f"Results: Model={m}, Skip={skip}, Parameter Index={index}, Train Loss={min(train_losses):.3f}, Test Loss={min(test_losses):.3f}"
        )
        results.append((m, skip, index, min(train_losses), min(test_losses)))
        try:
            dtype = [('m', 'U50'), ('skip', 'i4'), ('index', 'i4'), ('min_train_loss', 'f4'), ('min_test_loss', 'f4')]
            structured_array = np.array(results, dtype=dtype)
            np.savez('search_results_1000.npz', results=structured_array)
        except:
            print("Error Saving Search Results...")

    def get_top_5_indices(results_list):
        top_5_train = {}
        top_5_test = {}
        models = set(entry[0] for entry in results_list)  # Unique models

        for model in models:
            model_results = [entry for entry in results_list if entry[0] == model]
            # Sort by train loss and get top 5 indices
            train_sorted = sorted(model_results, key=lambda x: x[3])[:5]
            top_5_train[model] = [int(entry[2]) for entry in train_sorted]

            # Sort by test loss and get top 5 indices
            test_sorted = sorted(model_results, key=lambda x: x[4])[:5]
            top_5_test[model] = [int(entry[2]) for entry in test_sorted]

        return top_5_train, top_5_test

    skip_1_top_train, skip_1_top_test = get_top_5_indices(results)

    # Print results
    print("Top 5 indices for skip=1 (train loss):", skip_1_top_train)
    print("Top 5 indices for skip=1 (test loss):", skip_1_top_test)


if __name__ == "__main__":
    typer.run(main)
