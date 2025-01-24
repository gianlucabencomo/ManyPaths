import typer
import os
import re
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn

import learn2learn as l2l
import matplotlib.pyplot as plt

from utils import set_random_seeds
from visualize import plot_loss, plot_meta_test_results
from evaluation import evaluate, baseline_evaluate
from initialization import init_model, init_dataset

from scipy.stats import sem, t
import pandas as pd
from constants import *

def get_filenames_seeds_indices(directory, experiment, model, data_type, skip):
    # Construct the regex pattern
    if experiment == "mod":
        pattern = (
            rf"{experiment}_"  # Experiment name
            rf"{model}_"  # Model name
            r"(\d+)_"  # Capturing group for index
            rf"{data_type}_"  # Data type
            rf"{skip}_"  # Skip value
            r"(\d+)\.pth"  # Capturing group for seed
        )
    else:
        pattern = (
            rf"{experiment}_"  # Experiment name
            rf"{model}_"  # Model name
            r"(\d+)_"  # Capturing group for index
            rf"{data_type}_"  # Data type
            r"(\d+)\.pth"  # Capturing group for seed
        )

    results = []
    try:
        for filename in os.listdir(directory):
            match = re.match(pattern, filename)
            if match:
                index = match.group(1)  # Extracted index
                seed = match.group(2)  # Extracted seed
                filepath = os.path.join(directory, filename)
                results.append((filepath, int(index), int(seed)))
    except FileNotFoundError:
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    except PermissionError:
        raise PermissionError(f"Permission denied for accessing '{directory}'.")

    return results


def compute_mean_and_variance(results, experiment="mod"):
    if experiment == "mod":
        adapt_1 = np.array([[task["losses"][0] for task in res] for res in results])
        adapt_10 = np.array([[task["losses"][1] for task in res] for res in results])
        adapt_100 = np.array([[task["losses"][2] for task in res] for res in results])
    else:
        adapt_1 = np.array([[task["accuracies"][0] for task in res] for res in results])
        adapt_10 = np.array([[task["accuracies"][1] for task in res] for res in results])
        adapt_100 = np.array([[task["accuracies"][2] for task in res] for res in results])
    m = np.array([task["m"] for task in results[0]])

    mean_1 = np.mean(adapt_1, axis=0)  # Mean across seeds
    stderr_1 = sem(adapt_1, axis=0)  # Standard error of the mean
    var_1 = stderr_1**2  # Variance

    mean_10 = np.mean(adapt_10, axis=0)  # Mean across seeds
    stderr_10 = sem(adapt_10, axis=0)  # Standard error of the mean
    var_10 = stderr_10**2  # Variance

    mean_100 = np.mean(adapt_100, axis=0)  # Mean across seeds
    stderr_100 = sem(adapt_100, axis=0)  # Standard error of the mean
    var_100 = stderr_100**2  # Variance

    return mean_1, var_1, mean_10, var_10, mean_100, var_100, m


def main(
    directory: str = "./state_dicts/",
    output_folder: str = "results",
    experiment: str = "mod",  # ["mod", "concept", "omniglot"]
    test_seeds: int = 10,
    plot: bool = False,
    save: bool = False,
):
    architectures = ["mlp", "lstm", "cnn", "transformer"]
    if experiment == "omniglot":
        data_types = ["all", "ancient", "asian", "middle", "european"]
        n_supports = [5]
    else:
        data_types = ["image", "bits", "number"]
        n_supports = [20, 40, 100] if experiment == "mod" else [5, 10, 15]
    if experiment == "concept":
        data_types.remove("number")
    skips = [1, 2] if experiment == "mod" else [None]
    channels = 3 if experiment == "concept" else 1
    bits = 4 if experiment == "concept" else 8
    criterion = nn.MSELoss() if experiment == "mod" else nn.BCEWithLogitsLoss() if experiment == "concept" else nn.CrossEntropyLoss()
    
    results_master = []
    for m in architectures:
        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else ("cpu" if m in ["mlp", "lstm"] else "mps")
        )
        print(f"Device: {device}")
        for data_type in data_types:
            if data_type == "number" and m != "mlp":
                continue
            if data_type in ["number", "bits"] and m == "cnn":
                continue
            for skip in skips:
                models = get_filenames_seeds_indices(
                    directory, experiment, m, data_type, skip
                )
                for _, index, seed in models:
                    model = init_model(
                        m,
                        "image" if experiment == "omniglot" else data_type,
                        index=index,
                        verbose=True,
                        channels=channels,
                        bits=bits,
                        n_output=20 if experiment == "omniglot" else 1,
                    ).to(device)
    
                    vals = {n_support: [] for n_support in n_supports}
                    tests = {n_support: [] for n_support in n_supports}
                    for s in tqdm(range(seed, seed + test_seeds)):
                        set_random_seeds(s)
                        for n_support in n_supports:
                            test_dataset, val_dataset = init_dataset(
                                experiment, m, data_type, skip, n_support=n_support
                            )

                            # Evaluate on validation set
                            _, val_results = baseline_evaluate(
                                model,
                                val_dataset,
                                criterion,
                                device,
                                [1, 10, 100],
                                return_results=True,
                            )
                            vals[n_support].append(val_results)
                            # Optionally plot the validation results
                            if plot:
                                plot_meta_test_results(val_results)
                                plt.show()

                            # Evaluate on test set
                            _, test_results = baseline_evaluate(
                                model,
                                test_dataset,
                                criterion,
                                device,
                                [1, 10, 100],
                                return_results=True,
                            )
                            tests[n_support].append(test_results)
                            # Optionally plot the test results
                            if plot:
                                plot_meta_test_results(test_results)
                                plt.show()

                    for n_support in n_supports:
                        (
                            val_mean_1,
                            val_var_1,
                            val_mean_10,
                            val_var_10,
                            val_mean_100,
                            val_var_100,
                            val_m,
                        ) = compute_mean_and_variance(vals[n_support], experiment=experiment)
                        (
                            test_mean_1,
                            test_var_1,
                            test_mean_10,
                            test_var_10,
                            test_mean_100,
                            test_var_100,                            
                            test_m,
                        ) = compute_mean_and_variance(tests[n_support], experiment=experiment)

                        # For each task (m) in validation
                        for i, m_val in enumerate(val_m):
                            results_master.append(
                                {
                                    "experiment": experiment,
                                    "model": m,
                                    "data_type": data_type,
                                    "skip": skip,
                                    "n_support": n_support,
                                    "Stage": "Val",
                                    # The "task index" or moduli
                                    "m": m_val,
                                    # post-adaptation
                                    "mean_1": float(val_mean_1[i]),
                                    "var_1": float(val_var_1[i]),
                                    "mean_10": float(val_mean_10[i]),
                                    "var_10": float(val_var_10[i]),
                                    "mean_100": float(val_mean_100[i]),
                                    "var_100": float(val_var_100[i]),
                                    "index": index,
                                }
                            )

                        # For each task (m) in test
                        for i, m_test in enumerate(test_m):
                            results_master.append(
                                {
                                    "experiment": experiment,
                                    "model": m,
                                    "data_type": data_type,
                                    "skip": skip,
                                    "n_support": n_support,
                                    "Stage": "Test",
                                    "m": m_test,
                                    "mean_1": float(test_mean_1[i]),
                                    "var_1": float(test_var_1[i]),
                                    "mean_10": float(test_mean_10[i]),
                                    "var_10": float(test_var_10[i]),
                                    "mean_100": float(test_mean_100[i]),
                                    "var_100": float(test_var_100[i]),
                                    "index": index,
                                }
                            )
    df_master = pd.DataFrame(results_master)
    print(df_master)

    if save:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{experiment}_baseline.csv")
        df_master.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    typer.run(main)
