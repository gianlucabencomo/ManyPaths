from functools import partial
from typing import Callable, Any

import jax
import jax.numpy as jnp

import optax

from data import sample_tasks

PyTree = Any


def MAML(
    f: Callable,
    params: PyTree,
    alpha: float,
    beta: float,
    T: int,
    K: int,
    train_iter: int = 20000,
    steps: int = 1,
):
    """MAML training loop for Sine Wave Regression Task.

    Trains a Flax NN (f) with parameters (params) using adam as the outer loop optimizer
    and SGD as the inner loop optimizer. The inner loop loss has be specified as MSE.

    Args:
        f: Callable
            Flax neural network.
        params: PyTree
            Parameters of flax neural network.
        alpha: float
            Inner loop learning rate.
        beta: float
            Outer loop learning rate.
        T: int
            Number of tasks to sample per batch.
        K: int
            Number of datapoints to sample per task.
        train_iter: int
            Number of MAML training iterations.
        steps: int
            Number of optimization steps to take in the inner loop.

    Returns:
        params:
            Updated params post-MAML training.
        losses: list
            List of losses collected during training.

    Raises:
        None.
    """
    optimizer = optax.adam(
        learning_rate=beta,
    )

    @jax.jit
    def loss(params, X, y):
        """MSE Loss."""
        return jnp.mean((y - f(params, X)) ** 2)

    @jax.jit
    def inner_update(params, X, y):
        """MAML inner loop update."""

        def step(carry, _):
            i, params = carry
            g = jax.grad(loss)(params, X, y)
            params = jax.tree_map(lambda p, g: p - alpha * g, params, g)
            return (i + 1, params), None

        (_, params), _ = jax.lax.scan(step, (0, params), None, length=steps)

        return params

    @jax.jit
    def batch_maml_loss(params, X1, y1, X2, y2):
        """Vmapped MAML loss."""

        def maml_loss(params, X1, y1, X2, y2):
            params = inner_update(params, X1, y1)
            return loss(params, X2, y2)

        losses = jax.vmap(partial(maml_loss, params))(X1, y1, X2, y2)
        return jnp.mean(losses)

    @jax.jit
    def outer_update(params, X1, y1, X2, y2):
        """MAML outer loop update."""
        state = optimizer.init(params)

        def step(params, state, X1, y1, X2, y2):
            g = jax.grad(batch_maml_loss)(params, X1, y1, X2, y2)
            updates, state = optimizer.update(g, state)
            params = optax.apply_updates(params, updates)
            return params

        return step(params, state, X1, y1, X2, y2)

    losses = []
    for _ in range(train_iter):
        X1, y1, X2, y2 = sample_tasks(T, K)
        params = outer_update(params, X1, y1, X2, y2)
        losses.append(batch_maml_loss(params, X1, y1, X2, y2))

    return params, losses
