import typer
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import learn2learn as l2l
import matplotlib.pyplot as plt


from datasets import MetaModuloDataset
from models import MLP, CNN, LSTM, Transformer
from utils import set_random_seeds

MLP_PARAMS = [
    (8, 2),  # 0
    (16, 2),  # 1
    (32, 2),  # 2
    (64, 2),  # 3
    (8, 4),  # 4
    (16, 4),  # 5
    (32, 4),  # 6
    (64, 4),  # 7, (3)
    (8, 6),  # 8
    (16, 6),  # 9
    (32, 6),  # 10
    (64, 6),  # 11, (1)
    (8, 8),  # 12
    (16, 8),  # 13
    (32, 8),  # 14
    (64, 8),  # 15, (2)
]

CNN_PARAMS = [
    ([16, 8], 2),  # 0
    ([32, 16], 2),  # 1
    ([64, 32], 2),  # 2
    ([128, 64], 2),  # 3
    ([16, 8, 4, 2], 4),  # 4
    ([32, 16, 8, 4], 4),  # 5
    ([64, 32, 16, 8], 4),  # 6
    ([128, 64, 32, 16], 4),  # 7
    ([16, 8, 4, 2], 6),  # 8
    ([32, 16, 8, 4], 6),  # 9
    ([64, 32, 16, 8], 6),  # 10, (1)
    ([128, 64, 32, 16], 6),  # 11
    ([16, 8, 4, 2], 8),  # 12, (3)
    ([32, 16, 8, 4], 8),  # 13, (2)
    ([64, 32, 16, 8], 8),  # 14
    ([128, 64, 32, 16], 8),  # 15
]

LSTM_PARAMS = [
    (8, 1),  # 0
    (16, 1),  # 1
    (32, 1),  # 2
    (64, 1),  # 3
    (8, 2),  # 4 (2)
    (16, 2),  # 5
    (32, 2),  # 6, (1)
    (64, 2),  # 7
    (8, 3),  # 8
    (16, 3),  # 9, (3)
    (32, 3),  # 10
    (64, 3),  # 11
    (8, 4),  # 12
    (16, 4),  # 13
    (32, 4),  # 14
    (64, 4),  # 15
]

TRANSFORMER_PARAMS = [
    (8, 1),  # 0
    (16, 1),  # 1
    (32, 1),  # 2
    (64, 1),  # 3
    (8, 2),  # 4
    (16, 2),  # 5
    (32, 2),  # 6 (4)
    (64, 2),  # 7 (3)
    (8, 3),  # 8
    (16, 3),  # 9
    (32, 3),  # 10 (2)
    (64, 3),  # 11
    (8, 4),  # 12
    (16, 4),  # 13
    (32, 4),  # 14 (1)
    (64, 4),  # 15
]
models = ["mlp", "transformer", "lstm", "cnn"]
skips = [1]
indices = np.arange(16)


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
    train_loader,
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
        for i, (X_s, y_s, X_q, y_q) in enumerate(train_loader):
            X_s, y_s, X_q, y_q = (
                X_s.to(device),
                y_s.to(device),
                X_q.to(device),
                y_q.to(device),
            )
            optimizer.zero_grad()
            meta_loss = 0.0
            for t in range(tasks_per_meta_batch):
                # Adaptation on the support set
                learner = meta.clone()
                for _ in range(adaptation_steps):
                    support_pred = learner(X_s[t])
                    support_loss = criterion(support_pred, y_s[t])
                    learner.adapt(support_loss)

                # Evaluate on the query set
                query_pred = learner(X_q[t])
                query_loss = criterion(query_pred, y_q[t])
                meta_loss += query_loss

            meta_loss /= tasks_per_meta_batch
            meta_loss.backward()
            optimizer.step()

        train_loss = evaluate(
            meta, test_train_dataset, criterion, device, [0, adaptation_steps]
        )
        test_loss = evaluate(
            meta, test_dataset, criterion, device, [0, adaptation_steps]
        )
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return train_losses, test_losses


def init_dataset(model, data_type, n_samples_per_task, skip):
    train_dataset = MetaModuloDataset(
        n_samples_per_task=n_samples_per_task,
        skip=skip,
        train=True,
        data=data_type,
        model=model,
    )
    test_dataset = MetaModuloDataset(
        n_samples_per_task=n_samples_per_task,
        skip=skip,
        train=False,
        data=data_type,
        model=model,
    )
    test_train_dataset = MetaModuloDataset(
        n_tasks=20,
        n_samples_per_task=n_samples_per_task,
        skip=skip,
        train=True,
        data=data_type,
        model=model,
    )

    return train_dataset, test_dataset, test_train_dataset


def init_model(model, data_type, index):
    if data_type == "image":
        n_input = 1024 if model == "mlp" else 16
    elif data_type == "bits":
        n_input = 8 if model == "mlp" else 1
    elif data_type == "number":
        n_input = 1
    else:
        raise ValueError("Data Type unrecognized.")

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
            dim_feedforward=2 * n_hidden,
            num_layers=n_layers,
        )
    return model


def main(
    seed: int = 100,
    epochs: int = 10,
    tasks_per_meta_batch: int = 4,  # static
    adaptation_steps: int = 1,  # train [1] test [0, 1]
    inner_lr: float = 1e-3,
    outer_lr: float = 1e-3,
):
    data_type = "image"
    n_samples_per_task = 20
    # init dataset and model
    results = []
    for m in models:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("cpu" if m in ["mlp", "lstm"] else "mps")
        )
        train_dataset, test_dataset, test_train_dataset = init_dataset(m, data_type, n_samples_per_task, skips[0])
        train_loader = DataLoader(train_dataset, batch_size=tasks_per_meta_batch, shuffle=True, drop_last=True)
        for index in indices:
            start = time.time()
            set_random_seeds(seed)
            model = init_model(m, data_type, index)

            # init meta-learner, loss, and meta-optimizer
            model = model.to(device)
            meta = l2l.algorithms.MetaSGD(model, lr=inner_lr, first_order=False).to(
                device
            )
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(meta.parameters(), lr=outer_lr)

            # meta-training
            train_losses, test_losses = meta_train(
                meta,
                train_loader,
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
                f"Results: Model={m}, Skip={skips[0]}, Parameter Index={index}, Train Loss={min(train_losses):.3f}, Test Loss={min(test_losses):.3f} ({time.time() - start:.2f}s)"
            )
            results.append((m, skips[0], index, min(train_losses), min(test_losses)))
            try:
                dtype = [
                    ("m", "U50"),
                    ("skip", "i4"),
                    ("index", "i4"),
                    ("min_train_loss", "f4"),
                    ("min_test_loss", "f4"),
                ]
                structured_array = np.array(results, dtype=dtype)
                np.savez(
                    f"search_results_{epochs}_{seed}.npz", results=structured_array
                )
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
