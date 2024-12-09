import typer
import numpy as np
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial

import optax

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from archive.blr.model import RNNModel
from data import numpy_collate, SyntheticTask, SineWaveTask


class RNN:
    def __init__(self, model, params, alpha):
        self.model = model
        self.params = params
        self.optimizer = optax.adam(learning_rate=alpha)

    @partial(jit, static_argnums=0)
    def loss(self, params, X, y):
        prediction = self.model.apply({"params": params}, X)
        return jnp.mean(jnp.square(y - prediction))

    def train(self, trainloader, epochs: int = 50):
        losses = []
        opt_state = self.optimizer.init(self.params)
        for epoch in tqdm(range(epochs)):
            for (X_train, y_train), (X_val, y_val) in trainloader:
                grads = grad(self.loss)(self.params, X_train, y_train)
                grad_vals = jax.tree_util.tree_leaves(grads)
                max_grad = max([jnp.max(g) for g in grad_vals])
                min_grad = min([jnp.min(g) for g in grad_vals])
                print(max_grad)
                print(min_grad)
                updates, opt_state = self.optimizer.update(grads, opt_state)
                self.params = optax.apply_updates(self.params, updates)
            losses.append(self.loss(self.params, X_val, y_val))
        return np.array(losses)


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

    if task == "sine":
        train = SineWaveTask(
            seed=seed,
            n_episodes=n_episodes,
            n_support=n_support_train,
            n_query=n_query_train,
            sequential=True,
        )
        test = SineWaveTask(
            seed=seed + 1,
            n_episodes=10,
            n_support=n_support_test,
            n_query=n_query_test,
            sequential=True,
        )
    elif task == "synthetic":
        train = SyntheticTask(
            seed=seed,
            n_episodes=n_episodes,
            M=M,
            n_support=n_support_train,
            n_query=n_query_train,
            mixed=False,
            sequential=True,
        )
        test = SyntheticTask(
            seed=seed + 1,
            n_episodes=10,
            M=M,
            n_support=n_support_test,
            n_query=n_query_test,
            mixed=False,
            sequential=True,
        )
    else:
        raise ValueError("Task name not recognized.")
    trainloader = DataLoader(
        train, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate
    )
    testloader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=numpy_collate)

    # init model
    rnn = RNNModel()
    params = rnn.init(init_key, jax.random.normal(input_key, (1, 1, 1)))["params"]

    model = RNN(rnn, params, 1e-3)
    losses = model.train(trainloader)
    print(losses)

    # params = model.params
    # model = model.model
    # ypred = model.apply({"params": params}, X_train)
    # plt.plot(X_train.squeeze(), ypred.squeeze(), "r")
    # plt.plot(X_train.squeeze(), y_train.squeeze(), "b")
    # plt.show()


if __name__ == "__main__":
    typer.run(main)
