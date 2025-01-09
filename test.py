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
from evaluation import evaluate
from initialization import init_model, init_dataset

from scipy.stats import sem, t
import pandas as pd

def get_filenames_seeds_indices(directory, experiment, model, data_type, skip):
    # Construct the regex pattern
    pattern = (
        rf"{experiment}_"      # Experiment name
        rf"{model}_"           # Model name
        r"(\d+)_"              # Capturing group for index
        rf"{data_type}_"       # Data type
        rf"{skip}_"            # Skip value
        r"(\d+)\.pth"          # Capturing group for seed
    )

    results = []
    try:
        for filename in os.listdir(directory):
            match = re.match(pattern, filename)
            if match:
                index = match.group(1)  # Extracted index
                seed = match.group(2)   # Extracted seed
                filepath = os.path.join(directory, filename)
                results.append((filepath, int(index), int(seed)))
    except FileNotFoundError:
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    except PermissionError:
        raise PermissionError(f"Permission denied for accessing '{directory}'.")

    return results

def compute_mean_and_ci(results, confidence=0.95):
    pre_adapt = np.array([[task["losses"][0] for task in res] for res in results])
    post_adapt = np.array([[task["losses"][1] for task in res] for res in results])
    m = np.array([task["m"] for task in results[0]])

    mean_pre = np.mean(pre_adapt, axis=0)  # Mean across seeds
    stderr_pre = sem(pre_adapt, axis=0)  # Standard error of the mean
    h_pre = stderr_pre * t.ppf(
        (1 + confidence) / 2, pre_adapt.shape[0] - 1
    )  # CI margin

    # Compute mean and 95% CI for post-adaptation losses
    mean_post = np.mean(post_adapt, axis=0)  # Mean across seeds
    stderr_post = sem(post_adapt, axis=0)  # Standard error of the mean
    h_post = stderr_post * t.ppf(
        (1 + confidence) / 2, post_adapt.shape[0] - 1
    )  # CI margin

    return mean_pre, h_pre, mean_post, h_post, m


def main(
    directory: str = "./state_dicts/",
    output_folder: str = "results",
    experiment: str = "mod",  # ["mod", "concept"]
    test_seeds: int = 100,
    adaptation_steps: int = 1,
    plot: bool = False,
    save: bool = False,
):
    architectures = ["mlp", "cnn", "transformer"]
    n_supports = [20, 40, 100] if experiment == "mod" else [5, 10, 15]
    data_types = ['image', 'bits', 'number']
    skips = [1, 2] if experiment == "mod" else [None]
    channels = 3 if experiment == "concept" else 1
    bits = 4 if experiment == "concept" else 8
    criterion = nn.MSELoss() if experiment == "mod" else nn.BCEWithLogitsLoss()

    results_master = []
    for m in architectures:
        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else ("cpu" if m in ["mlp", "lstm"] else "mps")
        )
        print(f"Device: {device}")
        for data_type in data_types:
            if (data_type == "number" and m != "mlp"):
                continue
            if (data_type in ["number", "bits"] and m == "cnn"):
                continue
            for skip in skips:
                if experiment == "mod":
                    models = get_filenames_seeds_indices(directory, experiment, m, data_type, skip)
                else:
                    raise ValueError

  
                for (filepath, index, seed) in models:
                    base_model = init_model(
                        m, data_type, index=index, verbose=True, channels=channels, bits=bits
                    ).to(device)
                    meta = l2l.algorithms.MetaSGD(base_model, lr=1e-3, first_order=False).to(device)

                    # Load the saved .pth file
                    if not os.path.exists(filepath):
                        raise FileNotFoundError(f"Specified pth file not found: {filepath}")
                    meta.load_state_dict(
                        torch.load(filepath, map_location=device, weights_only=False)
                    )
                    print(f"Loaded meta-learner state from {filepath}")

                    vals = {n_support: [] for n_support in n_supports}
                    tests = {n_support: [] for n_support in n_supports}
                    for s in tqdm(range(seed, seed + test_seeds)):
                        set_random_seeds(s)
                        for n_support in n_supports:
                            test_dataset, val_dataset = init_dataset(
                                experiment, m, data_type, skip, n_support=n_support
                            )

                            # Evaluate on validation set
                            _, val_results = evaluate(
                                meta,
                                val_dataset,
                                criterion,
                                device,
                                [0, adaptation_steps],
                                return_results=True,
                            )
                            vals[n_support].append(val_results)
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
                            tests[n_support].append(test_results)
                            # Optionally plot the test results
                            if plot:
                                plot_meta_test_results(test_results)
                                plt.show()

                    for n_support in n_supports:
                        # Compute mean and 95% CI for validation and test losses
                        _, _, val_mean_post, val_h_post, val_m = compute_mean_and_ci(vals[n_support])
                        _, _, test_mean_post, test_h_post, test_m = compute_mean_and_ci(tests[n_support])

                        # For each task (m) in validation
                        for i, m_val in enumerate(val_m):
                            results_master.append({
                                "experiment": experiment,
                                "model": m,
                                "data_type": data_type,
                                "skip": skip,
                                "n_support": n_support,
                                "Stage": "Val",
                                # The "task index" or moduli
                                "m": m_val,
                                # post-adaptation
                                "mean_post": float(val_mean_post[i]),
                                "ci_post": float(val_h_post[i]),
                                "index": index,
                            })

                        # For each task (m) in test
                        for i, m_test in enumerate(test_m):
                            results_master.append({
                                "experiment": experiment,
                                "model": m,
                                "data_type": data_type,
                                "skip": skip,
                                "n_support": n_support,
                                "Stage": "Test",
                                "m": m_test,
                                "mean_post": float(test_mean_post[i]),
                                "ci_post": float(test_h_post[i]),
                                "index": index,
                            })
    df_master = pd.DataFrame(results_master)
    print(df_master.head())

    if save:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{experiment}.csv")
        df_master.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")



if __name__ == "__main__":
    typer.run(main)