import typer

import numpy as np
import matplotlib.pyplot as plt


def load_results(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data["arr_0"].tolist()  # Extract the saved array and convert back to a list


def main(
    skip: int = 1, epochs: int = 10000, train: bool = False, plot_pre: bool = False
):
    mlp_test_path = (
        f"./results/mlp_{'train' if train else 'test'}_skip_{skip}_epochs_{epochs}.npz"
    )
    cnn_test_path = (
        f"./results/cnn_{'train' if train else 'test'}_skip_{skip}_epochs_{epochs}.npz"
    )
    lstm_test_path = (
        f"./results/lstm_{'train' if train else 'test'}_skip_{skip}_epochs_{epochs}.npz"
    )
    transformer_test_path = f"./results/transformer_{'train' if train else 'test'}_skip_{skip}_epochs_{epochs}.npz"

    # Load results for all models
    mlp_test_results = load_results(mlp_test_path)
    from visualize import plot_meta_test_results

    cnn_test_results = load_results(cnn_test_path)
    lstm_test_results = load_results(lstm_test_path)
    transformer_test_results = load_results(transformer_test_path)
    # plot_meta_test_results(transformer_test_results)

    # Aggregate results for plotting
    results_by_model = {
        "MLP": mlp_test_results,
        "CNN": cnn_test_results,
        "LSTM": lstm_test_results,
        "Transformer": transformer_test_results,
    }

    # Initialize containers for aggregated losses
    pre_adaptation_losses = {model: {} for model in results_by_model.keys()}
    post_adaptation_losses = {model: {} for model in results_by_model.keys()}

    # Process results for each model
    for model, results in results_by_model.items():
        for task in results:
            m = task["m"]
            if m not in pre_adaptation_losses[model]:
                pre_adaptation_losses[model][m] = []
                post_adaptation_losses[model][m] = []
            pre_adaptation_losses[model][m].append(task["pre_adaptation_loss"])
            post_adaptation_losses[model][m].append(task["post_adaptation_loss"])

    colors = {
        "MLP": "blue",
        "CNN": "green",
        "LSTM": "orange",
        "Transformer": "red",
    }

    # Plot results
    plt.figure(figsize=(12, 6))
    for model in results_by_model.keys():
        ms = sorted(pre_adaptation_losses[model].keys())
        pre_losses = [np.mean(pre_adaptation_losses[model][m]) for m in ms]
        post_losses = [np.mean(post_adaptation_losses[model][m]) for m in ms]

        if plot_pre:
            plt.plot(
                ms,
                pre_losses,
                label=f"{model} (Pre-Adaptation)",
                linewidth=2,
                linestyle="dotted",
                marker="o",
                color=colors[model],
            )
        plt.plot(
            ms,
            post_losses,
            label=f"{model} (Post-Adaptation)",
            linewidth=2,
            linestyle="solid",
            marker="o",
            color=colors[model],
        )

    # Create custom legend for models
    model_legend_elements = [
        plt.Line2D([0], [0], color=color, lw=2, label=model)
        for model, color in colors.items()
    ]
    model_legend = plt.legend(
        handles=model_legend_elements, loc="upper left", title="Models"
    )

    # Create custom legend for pre-/post-adaptation
    if plot_pre:
        style_legend_elements = [
            plt.Line2D(
                [0], [0], color="black", lw=2, linestyle="--", label="Pre-Adaptation"
            ),
            plt.Line2D(
                [0], [0], color="black", lw=2, linestyle="-", label="Post-Adaptation"
            ),
        ]
        style_legend = plt.legend(
            handles=style_legend_elements, loc="upper right", title="Adaptation"
        )

    # Add both legends to the plot
    plt.gca().add_artist(model_legend)

    plt.title(
        f"{'Pre- and ' if plot_pre else ''}Post-Adaptation Losses for Each Modulus (Skip={skip}, Epochs={epochs})"
    )
    plt.xlabel("Modulus (m)")
    plt.xticks(ticks=ms, labels=ms)
    plt.ylabel("Mean Squared Error")
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    typer.run(main)
