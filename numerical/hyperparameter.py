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
    (16, 2),  # 0
    (32, 2),  # 1
    (64, 2),  # 2
    (128, 2), # 3
    (16, 4),  # 4
    (32, 4),  # 5
    (64, 4),  # 6
    (128, 4), # 7, (3)
    (16, 6),  # 8
    (32, 6),  # 9
    (64, 6),  # 10
    (128, 6), # 11, (2)
    (16, 8),  # 12
    (32, 8),  # 13
    (64, 8),  # 14
    (128, 8), # 15, (1)
    (16, 10), # 16
    (32, 10), # 17
    (64, 10), # 18
    (128, 10),# 19
]

CNN_PARAMS = [
    ([16, 8], 2),           # 0
    ([32, 16], 2),          # 1
    ([64, 32], 2),          # 2
    ([128, 64], 2),         # 3
    ([16, 8], 4),           # 4
    ([32, 16, 8], 4),       # 5
    ([64, 32, 16], 4),      # 6
    ([128, 64, 32], 4),     # 7
    ([16, 8], 6),           # 8
    ([32, 16, 8], 6),       # 9
    ([64, 32, 16], 6),      # 10, (3)
    ([128, 64, 32, 16], 6), # 11
    ([16, 8], 8),           # 12
    ([32, 16, 8], 8),       # 13, (2)
    ([64, 32, 16, 8], 8),   # 14
    ([128, 64, 32, 16], 8), # 15
    ([16, 8], 10),          # 16
    ([32, 16, 8], 10),      # 17, (1)
    ([64, 32, 16, 8], 10),  # 18
    ([128, 64, 32, 16], 10),# 19
]

LSTM_PARAMS = [
    (16, 1),  # 0
    (32, 1),  # 1
    (64, 1),  # 2
    (128, 1), # 3
    (16, 2),  # 4
    (32, 2),  # 5
    (64, 2),  # 6, (1)
    (128, 2), # 7
    (16, 3),  # 8
    (32, 3),  # 9, (3)
    (64, 3),  # 10
    (128, 3), # 11
    (16, 4),  # 12
    (32, 4),  # 13
    (64, 4),  # 14
    (128, 4), # 15
    (16, 5),  # 16
    (32, 5),  # 17, (2)
    (64, 5),  # 18
    (128, 5), # 19
]

TRANSFORMER_PARAMS = [
    (16, 1),  # 0
    (32, 1),  # 1
    (64, 1),  # 2
    (128, 1), # 3
    (16, 2),  # 4
    (32, 2),  # 5
    (64, 2),  # 6
    (128, 2), # 7
    (16, 3),  # 8
    (32, 3),  # 9
    (64, 3),  # 10
    (128, 3), # 11
    (16, 4),  # 12
    (32, 4),  # 13
    (64, 4),  # 14
    (128, 4), # 15
    (16, 5),  # 16
    (32, 5),  # 17
    (64, 5),  # 18
    (128, 5), # 19
]
models = ["transformer", "lstm", "cnn", "mlp"]
skips = [1]
indices = np.arange(20)


def evaluate(
    meta, dataset, criterion, device, adaptation_steps=[0, 1], return_results=False
):
    meta.train()
    meta_loss, results = 0.0, []
    for task in dataset.tasks:
        idx = torch.randint(0, len(task), size=(1,)).item()
        X_s, X_num_s, y_s, X_q, X_num_q, y_q, m = task[idx]
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
    test_train_dataset,
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
                meta, test_train_dataset, criterion, device, [0, adaptation_steps]
            )
            test_loss = evaluate(
                meta, test_dataset, criterion, device, [0, adaptation_steps]
            )
            # append
            train_losses.append(train_loss)
            test_losses.append(test_loss)

    return train_losses, test_losses


def init_dataset(model, skip):
    train_dataset = MetaImageModuloDataset(
        n_tasks=20,
        n_samples_per_task=20,
        range_max=100,
        n_samples=1000,
        skip=skip,
        train=True,
        model=model,
    )
    test_dataset = MetaImageModuloDataset(
        n_tasks=20,
        n_samples_per_task=20,
        range_max=100,
        n_samples=1,
        skip=skip,
        train=False,
        model=model,
    )
    test_train_dataset = MetaImageModuloDataset(
        n_tasks=20,
        n_samples_per_task=20,
        range_max=100,
        n_samples=1,
        skip=skip,
        train=True,
        model=model,
    )

    return train_dataset, test_dataset, test_train_dataset

def init_model(model, index):
    n_input = 1024 if model == "mlp" else 16
    if model == "mlp":
        n_hidden, n_layers = MLP_PARAMS[index]
        model = MLP(n_input=n_input, n_hidden=n_hidden, n_layers=n_layers)
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
            dim_feedforward=4*n_hidden,
            num_layers=n_layers,
        )
    return model

def main(
    seed: int = 100,
    epochs: int = 100,
    tasks_per_meta_batch: int = 4,  # static
    adaptation_steps: int = 1,  # train [1] test [0, 1]
    inner_lr: float = 1e-3,
    outer_lr: float = 1e-3,
):
    # init dataset and model
    results = []
    for m in models: 
        device = torch.device("cuda" if torch.cuda.is_available() else ("cpu" if m in ['mlp', 'lstm'] else 'mps'))
        train_dataset, test_dataset, test_train_dataset = init_dataset(m, skips[0])
        for index in indices:
            set_random_seeds(seed)
            model = init_model(m, index)

            # init meta-learner, loss, and meta-optimizer
            model = model.to(device)
            meta = l2l.algorithms.MetaSGD(model, lr=inner_lr, first_order=False).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(meta.parameters(), lr=outer_lr)

            # meta-training
            train_losses, test_losses = meta_train(
                meta,
                train_dataset,
                test_train_dataset,
                test_dataset,
                criterion,
                optimizer,
                device,
                epochs,
                tasks_per_meta_batch,
                adaptation_steps,
            )
            print(
                f"Results: Model={m}, Skip={skips[0]}, Parameter Index={index}, Train Loss={min(train_losses):.3f}, Test Loss={min(test_losses):.3f}"
            )
            results.append((m, skips[0], index, min(train_losses), min(test_losses)))
            try:
                dtype = [('m', 'U50'), ('skip', 'i4'), ('index', 'i4'), ('min_train_loss', 'f4'), ('min_test_loss', 'f4')]
                structured_array = np.array(results, dtype=dtype)
                np.savez(f'search_results_{epochs}_{seed}.npz', results=structured_array)
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
