import typer
import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import learn2learn as l2l
import matplotlib.pyplot as plt


from datasets import MetaBitConceptsDataset
from models import MLP, CNN, LSTM, Transformer
from utils import set_random_seeds
from visualize import plot_loss, plot_meta_test_results

MLP_PARAMS = (64, 6) # F
CNN_PARAMS = ([128, 64, 32, 16], 8) # F
LSTM_PARAMS = (64, 2)
TRANSFORMER_PARAMS = (64, 4) # F


def save_res(results, save_dir="results", file_prefix="meta_learning"):
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(f"{save_dir}/{file_prefix}.npz", results=results)
    print(f"Results saved to {save_dir}/{file_prefix}.npz")


def save_model(meta, save_dir="results", file_prefix="meta_learning"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(meta.state_dict(), f"{save_dir}/{file_prefix}.pth")
    print(f"Model saved to {save_dir}/{file_prefix}.pth")

def calculate_accuracy(predictions, targets):
    predictions = (predictions > 0.0).float()
    # Compare with targets and compute accuracy
    correct = (predictions == targets).sum().item()
    accuracy = correct / targets.numel()
    return accuracy

def evaluate(meta, dataset, criterion, device, adaptation_steps, return_results=False):
    meta.train()
    meta_loss, meta_acc, results = 0.0, [], []
    for task in dataset.tasks:
        X_s, X_num_s, y_s, X_q, X_num_q, y_q, m = task
        X_s, y_s, X_q, y_q = (
            X_s.to(device),
            y_s.to(device),
            X_q.to(device),
            y_q.to(device),
        )
        preds, losses, accs = [], [], []
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
                accs.append(calculate_accuracy(pred, y_q))
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
                "accuracies": accs,
            }
        )
        meta_acc.append(accs[:][1])
        meta_loss += losses[1]  # adaptation step = 1

    meta_loss /= len(dataset.tasks)
    meta_acc = np.mean(meta_acc)
    if return_results:
        return meta_loss, results
    else:
        return meta_loss, meta_acc


def meta_train(
    meta,
    train_loader,
    val_dataset,
    test_dataset,
    criterion,
    optimizer,
    device,
    epochs: int = 1000,
    tasks_per_meta_batch: int = 4,
    adaptation_steps: int = 1,
    patience = 20,  # Number of epochs to wait for improvement
):
    print("--- Meta-Training ---")
    meta.train()
    val_losses, test_losses = [], []
    best_val_loss = float("inf")
    best_model_state = None
    no_improve_epochs = 0
    for epoch in range(epochs):
        start = time.time()
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

        val_loss, val_acc = evaluate(
            meta, val_dataset, criterion, device, [0, adaptation_steps]
        )
        test_loss, test_acc = evaluate(
            meta, test_dataset, criterion, device, [0, adaptation_steps]
        )
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        no_improve_epochs += 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = meta.state_dict()
            no_improve_epochs = 0

        print(
            f"Epoch {epoch + 1}/{epochs}, Meta-Train Loss: {val_loss:.4f}, Meta-Test Loss: {test_loss:.4f}; Meta-Train Acc: {val_acc:.4f}, Meta-Test Acc: {test_acc:.4f} ({time.time() - start:.2f}s) {'*' if best_val_loss == val_loss else ''}"
        )
        if no_improve_epochs >= patience:
            print(f"No validation improvement after {patience} epochs. Stopping early...")
            break
        if np.isnan(val_loss):
            print(f"Meta-training diverged. Stopping early...")
            break
    return val_losses, test_losses, best_model_state

def init_dataset(model, data_type, n_samples_per_task):
    # init datasets
    train_dataset = MetaBitConceptsDataset(
        n_tasks=1000,
        n_samples_per_task=n_samples_per_task,
        data=data_type,
        model=model,
    )
    test_dataset = MetaBitConceptsDataset(
        n_tasks=100,
        n_samples_per_task=n_samples_per_task,
        data=data_type,
        model=model,
    )
    val_dataset = MetaBitConceptsDataset(
        n_tasks=20,
        n_samples_per_task=n_samples_per_task,
        data=data_type,
        model=model,
    )

    return train_dataset, test_dataset, val_dataset

def init_model(model, data_type):
    if data_type == "image":
        n_input = 32 * 32 * 3 if model == "mlp" else 16 * 3
    elif data_type == "bits":
        n_input = 4 if model == "mlp" else 1
    else:
        raise ValueError("Data Type unrecognized.")

    if model == "mlp":
        n_hidden, n_layers = MLP_PARAMS
        model = MLP(n_input=n_input, n_hidden=n_hidden, n_layers=n_layers)
    elif model == "cnn":
        n_hiddens, n_layers = CNN_PARAMS
        model = CNN(n_hiddens=n_hiddens, n_layers=n_layers)
    elif model == "lstm":
        n_hidden, n_layers = LSTM_PARAMS
        model = LSTM(n_input=n_input, n_hidden=n_hidden, n_layers=n_layers)
    elif model == "transformer":
        n_hidden, n_layers = TRANSFORMER_PARAMS
        model = Transformer(
            n_input=n_input,
            d_model=n_hidden,
            dim_feedforward=2 * n_hidden,
            num_layers=n_layers,
        )
    else:
        raise ValueError("Model unrecognized.")

    return model


def main(
    seed: int = 0,
    m: str = "mlp",  # ['mlp', 'cnn', 'lstm', 'transformer']
    data_type: str = "image",  # ['image', 'bits', 'number']
    n_samples_per_task: int = 9,
    epochs: int = 1000,  # or until convergence
    tasks_per_meta_batch: int = 4,  # static
    adaptation_steps: int = 1,  # train [1] test [0, 1]
    inner_lr: float = 1e-3,
    outer_lr: float = 1e-3,
    plot: bool = False,
    save: bool = False,
):
    set_random_seeds(seed)
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else ("cpu" if m in ["mlp", "lstm"] else "mps")
    )
    print(f"Device: {device}")

    # init dataset and model
    model = init_model(m, data_type)
    train_dataset, test_dataset, val_dataset = init_dataset(m, data_type, n_samples_per_task)
    
    train_loader = DataLoader(train_dataset, batch_size=tasks_per_meta_batch, shuffle=True, drop_last=True, pin_memory=(device == "cuda:0"))

    # init meta-learner, loss, and meta-optimizer
    model = model.to(device)
    meta = l2l.algorithms.MetaSGD(model, lr=inner_lr, first_order=False).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(meta.parameters(), lr=outer_lr)

    # meta-training
    train_losses, test_losses, state_dict = meta_train(
        meta,
        train_loader,
        val_dataset,
        test_dataset,
        criterion,
        optimizer,
        device,
        epochs,
        tasks_per_meta_batch,
        adaptation_steps,
    )
    if plot:
        plot_loss(train_losses, test_losses)

    # load best model
    meta.load_state_dict(state_dict)

    _, results = evaluate(
        meta,
        val_dataset,
        criterion,
        device,
        [0, adaptation_steps],
        return_results=True,
    )
    if plot:
        plot_meta_test_results(results)
    file_prefix = (
        m
        + "_train_"
        + data_type
        + "_"
        + str(n_samples_per_task)
        + "_"
        + str(adaptation_steps)
        + "_"
        + str(seed)
    )
    if save:
        #save_res(results, save_dir=m, file_prefix=file_prefix)
        save_model(meta, file_prefix=file_prefix)

    _, results = evaluate(
        meta,
        test_dataset,
        criterion,
        device,
        [0, adaptation_steps],
        return_results=True,
    )
    if plot:
        plot_meta_test_results(results)
    file_prefix = (
        m
        + "_test_"
        + data_type
        + "_"
        + str(n_samples_per_task)
        + "_"
        + str(adaptation_steps)
        + "_"
        + str(seed)
    )
    # if save:
    #     save_res(results, save_dir=m, file_prefix=file_prefix)

    if plot:
        plt.show()


if __name__ == "__main__":
    typer.run(main)
