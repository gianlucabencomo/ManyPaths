import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as np

from model import MLP

import matplotlib.pyplot as plt


def init_model(
    in_dim: tuple,
    layers: int = 2,
    hidden_dim: int = 40,
    key: jr.PRNGKey = jr.PRNGKey(0),
):
    "Sets up a basic MLP and returns its parameters plus a Callable."
    mlp = MLP(layers=layers, hidden_dim=hidden_dim)
    params = mlp.init(key, jnp.zeros(in_dim))

    @jax.jit
    def f(w, x):
        return mlp.apply(w, x)

    return f, params


def plot_inductive_bias(params: list, functions: list, labels: list = None):
    """Displays the inductive bias produced by each set of parameters for a list of functions."""
    X = np.linspace(-5, 5, 1000).reshape(-1, 1)
    plt.figure(figsize=(8, 8))
    linestyles = ["-", "--", ":", "-."]  # list of linestyles to cycle through
    for i, (p, f) in enumerate(zip(params, functions)):
        linestyle = linestyles[i % len(linestyles)]  # Cycle through linestyles
        label = f"Function {i}" if labels == None else labels[i]
        plt.plot(X, f(p, X), label=label, linestyle=linestyle)
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Inductive Biases")
    plt.show()


def set_random_seeds(seed):
    np.random.seed(seed)
