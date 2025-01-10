import typer

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import t

import pandas as pd

EPSILON = 1e-4

def summarize_experiment(
    df: pd.DataFrame,
    experiment: str,
):
    n_supports = sorted(df["n_support"].unique())
    stages = df["Stage"].unique()
    models = df["model"].unique()
    data_types = df["data_type"].unique()
    skips = df["skip"].unique()
    for skip in skips:
        for n_support in n_supports:
            for data_type in data_types:
                if data_type == "number":
                    continue
                for model in models:
                    if (data_type in ["number", "bits"]) and model == "cnn":
                        continue
                    for stage in stages:
                        if experiment == "concept":
                            subset = df[
                                (df["model"] == model) &
                                (df["n_support"] == n_support) &
                                (df["data_type"] == data_type) &
                                # (df["skip"] == skip) & 
                                (df["Stage"] == stage)
                            ]
                        else:
                            subset = df[
                                (df["model"] == model) &
                                (df["n_support"] == n_support) &
                                (df["data_type"] == data_type) &
                                (df["skip"] == skip) & 
                                (df["Stage"] == stage)
                            ]
                        if subset.empty:
                            continue
                        if stage == "Test" or experiment == "mod":
                            w = (1 / (subset["var_post"] + EPSILON))
                            w_norm = w / w.sum()
                            print(f"{stage}: Model = {model}, n_support = {n_support}, data type = {data_type}, Acc/Loss = {np.sum(w_norm * subset['mean_post']):.3f} +/- {1.96 * np.sqrt(1 / w.sum()):.3f}")
def main(directory: str = "results/", experiment: str = "concept"):
    csv_path = directory + experiment + ".csv"
    # Load DataFrame
    df = pd.read_csv(csv_path)

    # Quick check if the DataFrame is empty
    if df.empty:
        print(f"No data in {csv_path}!")
    else:
        summarize_experiment(df, experiment)

if __name__ == '__main__':
    typer.run(main)