import typer
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial

import optax

import torch
from torch.utils.data import DataLoader

from data import numpy_collate, SyntheticTask, SineWaveTask
from archive.blr.model import MLP


class MAML:
    def __init__(self, model, params, alpha=1e-2, beta=1e-3):
        """Initialize the MAML instance with model, parameters, and optimizers."""
        self.model = model
        self.params = params
        self.inner_optimizer = optax.sgd(learning_rate=alpha)
        self.outer_optimizer = optax.adam(learning_rate=beta)

    @partial(jit, static_argnums=0)
    def loss(self, params, X, y):
        """Calculate the mean squared error loss."""
        prediction = self.model.apply({"params": params}, X)
        return jnp.mean(jnp.square(y - prediction))

    @partial(jit, static_argnums=0)
    def fit_task(self, params, X, y, steps=1):
        """Perform the inner loop of MAML to adapt the parameters to a specific task."""
        opt_state = self.inner_optimizer.init(params)
        for _ in range(steps):
            grads = grad(self.loss)(params, X, y)
            updates, opt_state = self.inner_optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        return params

    @partial(jit, static_argnums=0)
    def maml_loss(self, params, X_s, y_s, X_q, y_q):
        """Compute the MAML loss across a batch of tasks."""
        adapted_params = vmap(self.fit_task, in_axes=(None, 0, 0))(params, X_s, y_s)
        losses = vmap(self.loss, in_axes=(0, 0, 0))(adapted_params, X_q, y_q)
        return jnp.mean(losses)

    @partial(jit, static_argnums=0)
    def fit_maml(self, params, X_s, y_s, X_q, y_q, steps=1):
        """Outer loop of MAML training to optimize across tasks."""
        opt_state = self.outer_optimizer.init(params)
        for _ in range(steps):
            grads = grad(self.maml_loss)(params, X_s, y_s, X_q, y_q)
            updates, opt_state = self.outer_optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        loss = self.maml_loss(params, X_s, y_s, X_q, y_q)
        return params, loss

    def train(self, trainloader):
        """Train the model using the MAML algorithm over a dataset with dynamic loss updates in tqdm."""
        losses = []
        pbar = tqdm(
            enumerate(trainloader),
            total=len(trainloader),
            desc="Starting MAML Training",
            leave=True,
        )

        for i, ((X_s, y_s), (X_q, y_q)) in pbar:
            self.params, loss = self.fit_maml(self.params, X_s, y_s, X_q, y_q)
            losses.append(loss)
            # Update tqdm bar with the current loss, dynamically displaying it.
            if i % 100 == 0 and i != 0:
                loss_avg = np.mean(losses[:-100])
                pbar.set_description(f"Meta-Loss (moving average): {loss_avg:.5f}")

        return losses

    def evaluate(self, testloader):
        """Evaluate the model using the test loader and plot results."""
        X, y, ypred, losses = [], [], [], []
        for (X_s, y_s), (X_q, y_q) in testloader:
            adapted_params = self.fit_task(self.params, X_s.squeeze(0), y_s.squeeze(0))
            loss = self.loss(adapted_params, X_q.squeeze(0), y_q.squeeze(0))
            X.append(X_q.squeeze(0))
            y.append(y_q.squeeze(0))
            ypred.append(self.model.apply({"params": adapted_params}, X_q.squeeze(0)))
            losses.append(loss)
        return np.array(X), np.array(y), np.array(ypred), losses

    def plot_results(self, X, y, y_pred):
        # Assuming X, y, and y_pred are of shape (batch_size, num_points, 1)
        batch_size, num_points, _ = X.shape
        cols = 2  # Number of columns in subplot grid
        rows = (batch_size + cols - 1) // cols  # Calculate required rows

        plt.figure(figsize=(18, rows * 2))  # Width of 15 inches and dynamic height
        for i in range(batch_size):
            ax = plt.subplot(rows, cols, i + 1)
            ax.plot(X[i, :, 0], y[i, :, 0], "b-", label="True")  # Flatten the last dim
            ax.plot(
                X[i, :, 0], y_pred[i, :, 0], "r--", label="Predicted"
            )  # Flatten the last dim
            ax.set_title(f"Task {i + 1}")
            ax.legend()
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        plt.tight_layout()
        plt.show()


def main(
    seed: int = 0,
    batch_size: int = 32,
    task: str = "sine",
    n_episodes: int = 2000000,
    n_support_train: int = 100,
    n_query_train: int = 100,
    n_support_test: int = 100,
    n_query_test: int = 100,
    M: int = 3,
    plot: bool = False,
):
    # set random seeds
    key = jax.random.PRNGKey(seed)
    key, input_key, init_key = jax.random.split(key, 3)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # meta training task + dataloader
    if task == "sine":
        train = SineWaveTask(
            seed=seed,
            n_episodes=n_episodes,
            n_support=n_support_train,
            n_query=n_query_train,
        )
        test = SineWaveTask(
            seed=seed + 1,
            n_episodes=10,
            n_support=n_support_test,
            n_query=n_query_test,
        )
    elif task == "synthetic":
        train = SyntheticTask(
            seed=seed,
            n_episodes=n_episodes,
            M=M,
            n_support=n_support_train,
            n_query=n_query_train,
            mixed=True,
        )
        test = SyntheticTask(
            seed=seed + 1,
            n_episodes=10,
            M=M,
            n_support=n_support_test,
            n_query=n_query_test,
            mixed=False,
        )
    else:
        raise ValueError("Task name not recognized.")
    trainloader = DataLoader(
        train, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate
    )
    testloader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=numpy_collate)

    # init model
    model = MLP()
    params = model.init(init_key, jax.random.normal(input_key, (1, 1)))["params"]

    # init maml + train
    maml = MAML(model, params)
    losses = maml.train(trainloader)

    if plot:
        X, y, y_pred, test_loss = maml.evaluate(testloader)
        print(f"Test Loss: {np.mean(test_loss):.5f}")
        maml.plot_results(X, y, y_pred)


if __name__ == "__main__":
    typer.run(main)
