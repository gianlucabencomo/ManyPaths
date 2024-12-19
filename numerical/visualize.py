import numpy as np
import torch
import matplotlib.pyplot as plt

STEPS = [0, 1]


def plot_loss(train_losses, test_losses, skip: int = 10):
    # Plot meta-training loss
    epochs = np.arange(len(train_losses)) * skip
    plt.plot(epochs, train_losses, label="Meta-Train Loss")
    plt.plot(epochs, test_losses, label="Meta-Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Meta Loss")
    plt.legend()


def plot_meta_test_results(results):
    n_tasks = len(results)
    cols = 5  # Number of columns in the subplot grid
    rows = (n_tasks + cols - 1) // cols  # Calculate number of rows required

    fig, axes = plt.subplots(rows, cols, figsize=(14, 2 * rows))
    axes = axes.flatten()

    for i, result in enumerate(results):
        ax = axes[i]
        m = result["m"]
        preds = []
        for pred in result["predictions"]:
            preds.append(pred.cpu().squeeze())
        X_s = result["X_s"].cpu().squeeze()
        y_s = result["y_s"].cpu().squeeze()
        X_q = result["X_q"].cpu().squeeze()
        y_q = result["y_q"].cpu().squeeze()

        sorted_indices = torch.argsort(X_q)
        ax.plot(
            X_q[sorted_indices],
            y_q[sorted_indices],
            label="True Function",
            linestyle="--",
        )
        ax.plot(
            X_s,
            y_s,
            ".",
            label="Support Points",
        )
        for i, step in enumerate(STEPS):
            ax.plot(
                X_q[sorted_indices], preds[i][sorted_indices], label=f"Steps {step}"
            )
        ax.set_title(f"m = {m}")

        if i % cols == 0:
            ax.set_ylabel("y = x mod m")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("x")

    # Remove any unused axes in the grid
    for j in range(n_tasks, len(axes)):
        fig.delaxes(axes[j])

    # Add a single legend overhead
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=cols,  # Adjust the number of columns in the legend
        fontsize="large",
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])


def plot_meta_test_losses(results):
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
