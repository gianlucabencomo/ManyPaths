import typer

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import t

STEPS = [0, 1]

def plot_loss(train_losses, test_losses):
    # Plot meta-training loss
    epochs = np.arange(len(train_losses))
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

xticks = {
    'mlp': ["2", "4", "6", "8"],
    'cnn': ["2", "4", "6", "8"],
    'lstm': ["1", "2", "3", "4"],
    'transformer': ["1", "2", "3", "4"]
}
yticks = {
    'mlp': ["8", "16", "32", "64"],
    'cnn': ["8", "16", "32", "64"],
    'lstm': ["8", "16", "32", "64"],
    'transformer': ["8", "16", "32", "64"]
}

def visualize_hyperparameter_search(file_paths):
    # Initialize storage for aggregated results
    aggregated_results = {}

    # Process each seed file
    for file_path in file_paths:
        seed = file_path.split("_")[-1].split(".")[0]  # Extract seed from the file name
        data = np.load(file_path, allow_pickle=True)
        results = data['results']

        # Group results by model type and index
        for entry in results:
            model = entry['m']
            index = entry['index']
            train_loss = entry['min_train_loss']
            
            if model not in aggregated_results:
                aggregated_results[model] = {}
            if index not in aggregated_results[model]:
                aggregated_results[model][index] = []
            
            aggregated_results[model][index].append(train_loss)

    # Create plots for the average values of the 5 seeds
    for model, results in aggregated_results.items():
        # Create a 4x4 matrix for mean train loss and confidence intervals
        mean_matrix = np.full((4, 4), np.nan)
        ci_matrix = np.full((4, 4), np.nan)
        for index, losses in results.items():
            row, col = divmod(index, 4)
            mean_matrix[row, col] = np.mean(losses)
            
            # Calculate 95% confidence interval
            if len(losses) > 1:  # CI is only meaningful with more than one sample
                sem = np.std(losses, ddof=1) / np.sqrt(len(losses))  # Standard error of the mean
                ci = t.ppf(0.975, len(losses) - 1) * sem  # 95% confidence interval
                ci_matrix[row, col] = ci
            else:
                ci_matrix[row, col] = 0  # If only one sample, CI is 0

        # Plot the 4x4 matrix for the model
        plt.figure(figsize=(8, 6))
        plt.imshow(mean_matrix, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=15)
        plt.colorbar(label='Average MSE')
        plt.title(f"{model.upper()} - Average of 5 Seeds with 95% CI")
        plt.xlabel('Parameter Index Column')
        plt.ylabel('Parameter Index Row')
        plt.xticks(range(4), xticks[model])
        plt.yticks(range(4), yticks[model])

        # Annotate each cell with mean ± CI
        for i in range(4):
            for j in range(4):
                mean_value = mean_matrix[i, j]
                ci_value = ci_matrix[i, j]
                if not np.isnan(mean_value):  # Only annotate valid cells
                    annotation = f"{mean_value:.2f} ± {ci_value:.2f}"
                    plt.text(
                        j,
                        i,
                        annotation,
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=10,
                    )

def main(base_path: str = "./search_results_10_"):
    seed_list = [100, 101, 102, 103, 104]
    file_paths = [f"{base_path}{seed}.npz" for seed in seed_list]  # Construct file paths
    visualize_hyperparameter_search(file_paths)
    plt.show()

if __name__ == '__main__':
    typer.run(main)