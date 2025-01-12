import typer

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import t

import pandas as pd

STEPS = [0, 1]

model_color = {
    "mlp": (0.1, 0.1, 0.8),  # Blue
    "cnn": (0.8, 0.1, 0.1),  # Red
    "lstm": (0.1, 0.8, 0.1),  # Green
    "transformer": (1.0, 0.5, 0.0),  # Orange
}
support_color = {
    20: (0.1, 0.1, 0.8),  # Blue
    40: (0.8, 0.1, 0.1),  # Red
    100: (0.1, 0.8, 0.1),  # Green
    5: (0.1, 0.1, 0.8),  # Blue
    10: (0.8, 0.1, 0.1),  # Red
    15: (0.1, 0.8, 0.1),  # Green
}
data_color = {
    "image": (0.1, 0.1, 0.8),  # Blue
    "bits": (0.8, 0.1, 0.1),  # Red
    "number": (0.1, 0.8, 0.1),  # Green
}
stage_marker = {
    "Val": "o",  # Circle for validation
    "Test": "*",  # Star for test
}


def plot_loss(train_losses, test_losses):
    """
    Plot meta-train and meta-test losses over epochs.
    """
    epochs = np.arange(len(train_losses))
    plt.plot(epochs, train_losses, label="Meta-Train Loss")
    plt.plot(epochs, test_losses, label="Meta-Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Meta Loss")
    plt.legend()


def plot_meta_test_results(results, n_tasks=20):
    """
    For a list of results (with inputs/predictions), create subplots
    visualizing pre- and post-adaptation predictions vs. the true function.
    """
    cols = 5
    rows = (n_tasks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2 * rows))
    axes = axes.flatten()

    i, ms = 0, []
    results = sorted(results, key=lambda x: x["m"])
    for result in results:
        m = result["m"]
        if m in ms:
            continue
        ms.append(m)
        ax = axes[i]

        preds = [pred.cpu().squeeze() for pred in result["predictions"]]
        X_s = result["X_s"].cpu().squeeze()
        y_s = result["y_s"].cpu().squeeze()
        X_q = result["X_q"].cpu().squeeze()
        y_q = result["y_q"].cpu().squeeze()

        sorted_indices = torch.argsort(X_q)
        # True function
        ax.plot(
            X_q[sorted_indices],
            y_q[sorted_indices],
            label="True Function",
            linestyle="--",
        )
        # Support points
        ax.plot(X_s, y_s, ".", label="Support Points")
        # Post-adaptation steps
        for k, step in enumerate(STEPS):
            ax.plot(
                X_q[sorted_indices], preds[k][sorted_indices], label=f"Steps {step}"
            )

        ax.set_title(f"m = {m}")
        if i % cols == 0:
            ax.set_ylabel("y = x mod m")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("x")
        i += 1

    # Remove any unused axes
    for j in range(n_tasks, len(axes)):
        fig.delaxes(axes[j])

    # Single legend overhead
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=cols, fontsize="large", frameon=False
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])


def plot_meta_test_losses(results):
    """
    Simple line plots of Pre-Adaptation vs. Post-Adaptation losses across tasks.
    """
    plt.figure()
    ms = []
    pre_loss = []
    post_loss = []
    for result in results:
        ms.append(result["m"])
        pre_loss.append(result["pre_adaptation_loss"])
        post_loss.append(result["post_adaptation_loss"])

    plt.plot(ms, pre_loss, "o-", label="Pre-Adaptation")
    plt.plot(ms, post_loss, "o-", label="Post-Adaptation")
    plt.legend()
    plt.tight_layout()


def plot_subset(
    subset: pd.DataFrame, group_label: str, color: tuple, marker: str, stage: str
):
    """
    A small helper function to scatter-plot subset data and then plot
    a line over mean values grouped by 'm'.
    """
    mean_over_means = subset.groupby("m")["mean_post"].mean()
    # Scatter
    plt.scatter(
        subset["m"],
        subset["mean_post"],
        label=group_label,
        color=color,
        marker=marker,
        alpha=0.6,
        s=50,
    )
    # Line plot
    plt.plot(
        mean_over_means.index,
        mean_over_means.values,
        color=color,
        linestyle="--" if stage == "Val" else "-",
        alpha=0.8,
    )


# ===============================
# 1) Plot by data type and stage
# ===============================
def plot_by_data_type_and_stage(
    df: pd.DataFrame,
    wlims: bool = False,
):
    """
    For each combination of skip -> model -> n_support, we create a figure
    and plot the data by data_type, distinguishing stage by marker style.
    Optionally, set xlims and ylims.
    """
    n_supports = sorted(df["n_support"].unique())
    stages = df["Stage"].unique()
    models = df["model"].unique()
    data_types = df["data_type"].unique()
    skips = df["skip"].unique()
    xlims, ylims = None, None
    for skip in skips:
        if wlims:
            xlims = (-0.1, 20.1) if skip == 1 else None
        ylims = (-0.5, 50.5) if skip == 2 else (-0.1, 10.1) if wlims else None
        for model in models:
            if model == "cnn":
                continue
            for n_support in n_supports:
                plt.figure(figsize=(10, 6))
                for data_type in data_types:
                    for stage in stages:
                        subset = df[
                            (df["model"] == model)
                            & (df["n_support"] == n_support)
                            & (df["data_type"] == data_type)
                            & (df["skip"] == skip)
                            & (df["Stage"] == stage)
                        ]
                        if not subset.empty:
                            group_label = f"Meta-{stage}, {data_type.capitalize()}"
                            plot_subset(
                                subset=subset,
                                group_label=group_label,
                                color=data_color[data_type],
                                marker=stage_marker[stage],
                                stage=stage,
                            )
                if xlims is not None:
                    plt.xlim(xlims)
                if ylims is not None:
                    plt.ylim(ylims)

                plt.xlabel("Moduli (m)")
                plt.ylabel("Mean Squared Error (MSE)")
                plt.title(
                    f"{'Odd/Even' if skip == 2 else '20/20'}, {model.capitalize()}, Num. Support = {n_support}"
                )
                plt.legend(title="Data Split, Data Representation")
                plt.grid(alpha=0.2)
                plt.tight_layout()
                if xlims == None and ylims == None:
                    plt.savefig(
                        f"figures/data_comp_{skip}_{model}_{n_support}.pdf",
                        format="pdf",
                        dpi=300,
                        bbox_inches="tight",
                    )
                else:
                    plt.savefig(
                        f"figures/data_comp_{skip}_{model}_{n_support}_w_lims.pdf",
                        format="pdf",
                        dpi=300,
                        bbox_inches="tight",
                    )


# ===============================
# 2) Plot by support size & stage
# ===============================
def plot_by_support_and_stage(
    df: pd.DataFrame,
    wlims: bool = False,
):
    """
    For each combination of skip -> model -> data_type, we create a figure
    and plot the data by n_support, distinguishing stage by marker style.
    Optionally, set xlims and ylims.
    """
    n_supports = sorted(df["n_support"].unique())
    stages = df["Stage"].unique()
    models = df["model"].unique()
    data_types = df["data_type"].unique()
    skips = df["skip"].unique()
    xlims, ylims = None, None
    for skip in skips:
        if wlims:
            xlims = (-0.1, 20.1) if skip == 1 else None
        ylims = (-0.5, 50.5) if skip == 2 else (-0.1, 10.1) if wlims else None
        for model in models:
            for data_type in data_types:
                # Skip invalid combos if needed
                if data_type == "number" and model != "mlp":
                    continue
                if (data_type in ["number", "bits"]) and model == "cnn":
                    continue

                plt.figure(figsize=(10, 6))
                for n_support in n_supports:
                    for stage in stages:
                        subset = df[
                            (df["model"] == model)
                            & (df["n_support"] == n_support)
                            & (df["data_type"] == data_type)
                            & (df["skip"] == skip)
                            & (df["Stage"] == stage)
                        ]
                        if not subset.empty:
                            group_label = f"Meta-{stage}, {n_support}"
                            plot_subset(
                                subset=subset,
                                group_label=group_label,
                                color=support_color[n_support],
                                marker=stage_marker[stage],
                                stage=stage,
                            )
                if xlims is not None:
                    plt.xlim(xlims)
                if ylims is not None:
                    plt.ylim(ylims)

                plt.xlabel("Moduli (m)")
                plt.ylabel("Mean Squared Error (MSE)")
                plt.legend(title="Data Split, Support Set Size")
                plt.title(
                    f"{'Odd/Even' if skip == 2 else '20/20'}, {model.capitalize()}, {data_type.capitalize()}"
                )
                plt.grid(alpha=0.2)
                plt.tight_layout()
                if xlims == None and ylims == None:
                    plt.savefig(
                        f"figures/support_comp_{skip}_{model}_{data_type}.pdf",
                        format="pdf",
                        dpi=300,
                        bbox_inches="tight",
                    )
                else:
                    plt.savefig(
                        f"figures/support_comp_{skip}_{model}_{data_type}_w_lims.pdf",
                        format="pdf",
                        dpi=300,
                        bbox_inches="tight",
                    )


# ===============================
# 3) Plot by model & stage
# ===============================
def plot_by_model_and_stage(
    df: pd.DataFrame,
    wlims: bool = False,
):
    """
    For each combination of skip -> n_support -> data_type, we create a figure
    and plot the data by model, distinguishing stage by marker style.
    Optionally, set xlims and ylims.
    """
    n_supports = sorted(df["n_support"].unique())
    stages = df["Stage"].unique()
    models = df["model"].unique()
    data_types = df["data_type"].unique()
    skips = df["skip"].unique()
    xlims, ylims = None, None
    for skip in skips:
        if wlims:
            xlims = (-0.1, 20.1) if skip == 1 else None
        ylims = (-0.5, 50.5) if skip == 2 else (-0.1, 10.1) if wlims else None
        for n_support in n_supports:
            for data_type in data_types:
                if data_type == "number":
                    continue
                plt.figure(figsize=(10, 6))
                for model in models:
                    if (data_type in ["number", "bits"]) and model == "cnn":
                        continue
                    for stage in stages:
                        subset = df[
                            (df["model"] == model)
                            & (df["n_support"] == n_support)
                            & (df["data_type"] == data_type)
                            & (df["skip"] == skip)
                            & (df["Stage"] == stage)
                        ]
                        if not subset.empty:
                            group_label = f"Meta-{stage}, {model.capitalize()}"
                            plot_subset(
                                subset=subset,
                                group_label=group_label,
                                color=model_color[model],
                                marker=stage_marker[stage],
                                stage=stage,
                            )
                if xlims is not None:
                    plt.xlim(xlims)
                if ylims is not None:
                    plt.ylim(ylims)

                plt.xlabel("Moduli (m)")
                plt.ylabel("Mean Squared Error (MSE)")
                plt.legend(title="Dataset Split, Model")
                plt.title(
                    f"{'Odd/Even' if skip == 2 else '20/20'}, {data_type.capitalize()}, Num. Support = {n_support}"
                )
                plt.grid(alpha=0.2)
                plt.tight_layout()
                if xlims == None and ylims == None:
                    plt.savefig(
                        f"figures/model_comp_{skip}_{n_support}_{data_type}.pdf",
                        format="pdf",
                        dpi=300,
                        bbox_inches="tight",
                    )
                else:
                    plt.savefig(
                        f"figures/model_comp_{skip}_{n_support}_{data_type}_w_lims.pdf",
                        format="pdf",
                        dpi=300,
                        bbox_inches="tight",
                    )


def main(directory: str = "results/", experiment: str = "mod"):
    csv_path = directory + experiment + ".csv"
    # Load DataFrame
    df = pd.read_csv(csv_path)
    if experiment == "concept":
        print(df)
        exit()

    # Quick check if the DataFrame is empty
    if df.empty:
        print(f"No data in {csv_path}!")
    else:
        plot_by_data_type_and_stage(df)
        plot_by_support_and_stage(df)
        plot_by_model_and_stage(df)
        if experiment == "mod":
            plot_by_data_type_and_stage(df, wlims=True)
            plot_by_support_and_stage(df, wlims=True)
            plot_by_model_and_stage(df, wlims=True)


if __name__ == "__main__":
    typer.run(main)
