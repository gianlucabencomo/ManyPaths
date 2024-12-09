import typer

from helper import init_model, set_random_seeds, plot_inductive_bias
from data import sample_tasks
from maml import MAML, fast_adaptation


def main(
    seed: int = 0,  # random seed for PRNG.
    alpha: float = 1e-1,  # inner loop learning rate.
    beta: float = 1e-3,  # outer loop learning rate.
    T: int = 4,  # number of tasks to sample per batch.
    K: int = 20,  # number of datapoints per task.
    train_iter: int = 2000,  # number of outer loop training steps.
    steps: int = 1,  # number of inner loop training steps.
    plot: bool = False,
):
    """Train and test original MAML Sine Wave Task."""
    # set random seed
    set_random_seeds(seed)

    in_dim = (1, 1)

    # init model 1
    f1, params1 = init_model(in_dim, layers=2, hidden_dim=10)
    # run maml with model 1 and return meta-learned params
    params1, _ = MAML(f1, params1, alpha, beta, T, K, train_iter, steps)
    X1, y1, X2, y2 = sample_tasks(T, K)
    print(X1.shape)
    params_ = fast_adaptation(f1, params1, X1[0], y1[0], alpha, steps)
    ypred = f1(params_, X2[0])
    print(ypred)
    import matplotlib.pyplot as plt

    plt.plot(X2[0], ypred, "ro")
    plt.plot(X2[0], y2[0], "bo")
    import numpy as np

    plt.show()
    exit()

    # init model 2
    f2, params2 = init_model(in_dim, layers=2, hidden_dim=100)
    # run maml with model 2 and return meta-learned params
    params2, _ = MAML(f2, params2, alpha, beta, T, K, train_iter, steps)

    # init model 3
    f3, params3 = init_model(in_dim, layers=2, hidden_dim=1000)
    # run maml with model 3 and return meta-learned params
    params3, _ = MAML(f3, params3, alpha, beta, T, K, train_iter, steps)

    if plot:
        labels = [
            "Layers=2, Hidden=10",
            "Layers=2, Hidden=100",
            "Layers=2, Hidden=1000",
        ]
        plot_inductive_bias([params1, params2, params3], [f1, f2, f3], labels=labels)


if __name__ == "__main__":
    typer.run(main)
