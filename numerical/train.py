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


def evaluate(meta, dataset, criterion, device, adaptation_steps, return_results=False):
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
    print("--- Meta-Training ---")
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
            print(
                f"Epoch {epoch + 1}/{epochs}, Meta-Train Loss: {train_loss:.4f}, Meta-Test Loss: {test_loss:.4f}"
            )

    return train_losses, test_losses


def init_dataset_and_model(
    model, model_size, data_type, n_tasks, n_samples_per_task, skip
):
    if data_type == "image":
        Dataset = MetaImageModuloDataset
        n_input = 1024 if model == "mlp" else 16
    elif data_type == "bits":
        Dataset = MetaBitStringModuloDataset
        n_input = 8 if model == "mlp" else 1
    else:
        Dataset = MetaNumberModuloDataset
        n_input = 1

    # init datasets
    train_dataset = Dataset(
        n_tasks, n_samples_per_task, range_max=100, skip=skip, train=True, model=model
    )
    test_dataset = Dataset(
        n_tasks, n_samples_per_task, range_max=100, skip=skip, train=False, model=model
    )
    test_train_dataset = Dataset(
        n_tasks, n_samples_per_task, range_max=100, skip=skip, train=True, model=model
    )

    if model_size == "fixed":
        if model == "mlp":
            model = MLP(n_input=n_input)
        elif model == "cnn":
            model = CNN()
        elif model == "lstm":
            model = LSTM(n_input=n_input)
        else:
            model = Transformer(n_input=n_input)
    else:
        raise NotImplementedError

    return train_dataset, test_dataset, test_train_dataset, model


def main(
    seed: int = 0,
    model: str = "mlp",  # ['mlp', 'cnn', 'lstm', 'transformer']
    model_size: str = "fixed",  # ['fixed', 'performance'] # TODO: set performance and fixed
    data_type: str = "image",  # ['image', 'bits', 'number']
    n_tasks: int = 20,  # static
    n_samples_per_task: int = 20,  # [20, 50, 100]
    epochs: int = 10000,  # or until convergence
    tasks_per_meta_batch: int = 4,  # static
    adaptation_steps: int = 1,  # train [1] test [0, 1]
    inner_lr: float = 1e-3,
    outer_lr: float = 1e-3,
    skip: int = 1,  # skip in [0, 1, 2] where 0 is train mod m
):
    set_random_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init dataset and model
    train_dataset, test_dataset, test_train_dataset, model = init_dataset_and_model(
        model, model_size, data_type, n_tasks, n_samples_per_task, skip
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
    plot_loss(train_losses, test_losses)

    _, results = evaluate(
        meta,
        test_train_dataset,
        criterion,
        device,
        [0, adaptation_steps],
        return_results=True,
    )
    plot_meta_test_results(results)
    data_path = (
        model
        + "_train_"
        + data_type
        + "_"
        + str(n_samples_per_task)
        + "_"
        + str(adaptation_steps)
        + "_"
        + str(skip)
        + "_"
        + str(seed)
        + ".npz"
    )
    # TODO: add save

    _, results = evaluate(
        meta,
        test_dataset,
        criterion,
        device,
        [0, adaptation_steps],
        return_results=True,
    )
    plot_meta_test_results(results)
    data_path = (
        model
        + "_test_"
        + data_type
        + "_"
        + str(n_samples_per_task)
        + "_"
        + str(adaptation_steps)
        + "_"
        + str(skip)
        + "_"
        + str(seed)
        + ".npz"
    )
    # TODO: add save

    plt.show()


if __name__ == "__main__":
    typer.run(main)
