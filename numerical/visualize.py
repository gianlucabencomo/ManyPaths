import torch
import matplotlib.pyplot as plt

def plot_loss(losses):
    # Plot meta-training loss
    plt.plot(range(len(losses)), losses, label="Meta Loss")
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
        pred_no_adap = result["pred_no_adap"].cpu().squeeze()
        pred_w_adap = result["pred_w_adap"].cpu().squeeze()
        X_q = result["X_q"].cpu().squeeze()
        y_q = result["y_q"].cpu().squeeze()

        sorted_indices = torch.argsort(X_q)
        ax.plot(X_q[sorted_indices], y_q[sorted_indices], label="True Function", linestyle='--')
        ax.plot(X_q[sorted_indices], pred_no_adap[sorted_indices], label="Pre-Adaptation")
        ax.plot(X_q[sorted_indices], pred_w_adap[sorted_indices], label="Post-Adaptation")
        ax.set_title(f"m = {m}")

        if i % cols == 0:
            ax.set_ylabel("x mod m")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("x")
        ax.legend()

    # Remove any unused axes in the grid
    for j in range(n_tasks, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()