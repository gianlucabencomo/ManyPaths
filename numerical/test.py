import typer
import os
import time
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import learn2learn as l2l
import matplotlib.pyplot as plt

from datasets import MetaModuloDataset
from models import MLP, CNN, LSTM, Transformer
from utils import set_random_seeds
from visualize import plot_loss, plot_meta_test_results

MLP_PARAMS = (64, 6)  # For MLP
CNN_PARAMS = ([128, 64, 32, 16], 8)  # For CNN
LSTM_PARAMS = (64, 2)  # For LSTM
TRANSFORMER_PARAMS = (64, 4)  # For Transformer

from scipy.stats import sem, t
import pandas as pd

model_color = {
    "mlp": [[(0.1, 0.1, 0.8), 1.0], [(0.1, 0.1, 0.8), 1.0]],
    "cnn": [[(0.1, 0.1, 0.8), 1.0], [(0.1, 0.1, 0.8), 1.0]], 
    "lstm": [[(0.1, 0.1, 0.8), 1.0], [(0.1, 0.1, 0.8), 1.0]],
    "transformer": [[(0.1, 0.1, 0.8), 1.0], [(0.1, 0.1, 0.8), 1.0]]
}

def plot_results(dfs):
    """
    Create a scatter plot with m on the x-axis, MSE on the y-axis,
    and points colored by stage ('Val' or 'Test').

    Parameters:
    - dfs: List of DataFrames containing results for each seed.
    """
    combined_df = pd.concat(dfs, ignore_index=True)  # Combine all seed DataFrames

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    for stage, group in combined_df.groupby("Stage"):
        plt.scatter(
            group["m"],
            group["Post Mean"],
            label=stage,
            color=model_color['mlp'][0][0],
            alpha=model_color['mlp'][0][1],
            s=50,
        )

    # Plot settings
    plt.title("MSE by Task Identifier (m) and Stage")
    plt.xlabel("Task Identifier (m)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend(title="Stage")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Show the plot
    plt.show()

def create_results_df(mean_post, h_post, m, stage):
    data = {
        "m": m,  # Flatten the task identifiers
        "Stage": [stage] * len(mean_post),  # Stage (Validation or Test)
        "Post Mean": mean_post,
        "± CI": h_post,
    }
    return pd.DataFrame(data)

def compute_mean_and_ci(results, confidence=0.95):
    pre_adapt = np.array([[task['losses'][0] for task in res] for res in results])
    post_adapt = np.array([[task['losses'][1] for task in res] for res in results])
    m = np.array([task['m'] for task in results[0]])

    mean_pre = np.mean(pre_adapt, axis=0)  # Mean across seeds
    stderr_pre = sem(pre_adapt, axis=0)  # Standard error of the mean
    h_pre = stderr_pre * t.ppf((1 + confidence) / 2, pre_adapt.shape[0] - 1)  # CI margin

    # Compute mean and 95% CI for post-adaptation losses
    mean_post = np.mean(post_adapt, axis=0)  # Mean across seeds
    stderr_post = sem(post_adapt, axis=0)  # Standard error of the mean
    h_post = stderr_post * t.ppf((1 + confidence) / 2, post_adapt.shape[0] - 1)  # CI margin


    return mean_pre, h_pre, mean_post, h_post, m


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


def init_dataset(model, data_type, n_samples_per_task, skip):
    test_dataset = MetaModuloDataset(
        n_samples_per_task=n_samples_per_task,
        skip=skip,
        train=False,
        data=data_type,
        model=model,
    )
    val_dataset = MetaModuloDataset(
        n_tasks=20,
        n_samples_per_task=n_samples_per_task,
        skip=skip,
        train=True,
        data=data_type,
        model=model,
    )
    return test_dataset, val_dataset


def init_model(model, data_type):
    """
    Initialize the base model (MLP, CNN, LSTM, or Transformer) with the correct input dimensions.
    """
    if data_type == "image":
        n_input = 1024 if model == "mlp" else 16
    elif data_type == "bits":
        n_input = 8 if model == "mlp" else 1
    elif data_type == "number":
        n_input = 1
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
    directory: str = "./results/",
    test_seeds: int = 100,
    m: str = 'mlp',
    data_type: str = "image",  # ['image', 'bits', 'number']
    n_samples_per_task: int = 20,
    adaptation_steps: int = 1,
    skip: int = 1,
    plot: bool = False,
    plot_final: bool = False,
    save: bool = False,
):
    seeds = [0, 1, 2, 3, 4]
    # Set seeds and device
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else ("cpu" if m in ["mlp", "lstm"] else "mps")
    )
    print(f"Device: {device}")

    base_model = init_model(m, data_type).to(device)
    meta = l2l.algorithms.MetaSGD(base_model, lr=1e-3, first_order=False).to(device)
    criterion = nn.MSELoss()

    dfs = []
    for seed in seeds:
        filename = directory + (
            m
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
            + ".pth"
        )

        # Load the saved .pth file
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Specified pth file not found: {filename}")
        meta.load_state_dict(torch.load(filename, map_location=device, weights_only=False))
        print(f"Loaded meta-learner state from {filename}")

        vals, tests = [], []
        for s in tqdm(range(seed, seed + test_seeds)):
            set_random_seeds(s)
            test_dataset, val_dataset = init_dataset(m, data_type, n_samples_per_task, skip)

            # Evaluate on validation set
            _, val_results = evaluate(
                meta,
                val_dataset,
                criterion,
                device,
                [0, adaptation_steps],
                return_results=True,
            )
            vals.append(val_results)
            # Optionally plot the validation results
            if plot:
                plot_meta_test_results(val_results)
                plt.show()

            # Evaluate on test set
            _, test_results = evaluate(
                meta,
                test_dataset,
                criterion,
                device,
                [0, adaptation_steps],
                return_results=True,
            )
            tests.append(test_results)
            # Optionally plot the test results
            if plot:
                plot_meta_test_results(test_results)
                plt.show()

        # Compute mean and 95% CI for validation and test losses
        _, _, val_mean_post, val_h_post, val_m = compute_mean_and_ci(vals)
        _, _, test_mean_post, test_h_post, test_m = compute_mean_and_ci(tests)

        val_df = create_results_df(val_mean_post, val_h_post, val_m, "Val")
        test_df = create_results_df(test_mean_post, test_h_post, test_m, "Test")

        # Combine validation and test DataFrames
        results_df = pd.concat([val_df, test_df], ignore_index=True)

        # Display the DataFrame
        print(results_df)

        dfs.append(results_df)

    if plot_final:
        plot_results(dfs)


if __name__ == "__main__":
    typer.run(main)
