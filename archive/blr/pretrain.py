import typer
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt

import optax
from flax.training import train_state

from torch.utils.data import DataLoader

from data import numpy_collate, SyntheticTask
from archive.blr.model import MLP


class PreTrain:
    def __init__(self, model, params, optimizer, loss) -> None:
        self.state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer
        )
        self.model = model.bind(self.state.params)
        self.loss = loss

    def calculate_loss(self, state, params, batch):
        X, y = batch
        logits = state.apply_fn(params, X).squeeze(axis=-1)
        loss = jnp.mean(self.loss(logits, y.squeeze()))
        return loss

    def train_model(self, data_loader, num_epochs=100):
        @jax.jit  # Jit the function for efficiency
        def _train_step(state, batch):
            # Gradient function
            grad_fn = jax.value_and_grad(
                self.calculate_loss,  # Function to calculate the loss
                argnums=1,  # Parameters are second argument of the function
                has_aux=False,
            )
            # Determine gradients for current model, parameters and batch
            loss, grads = grad_fn(state, state.params, batch)
            # Perform parameter update with gradients and optimizer
            state = state.apply_gradients(grads=grads)
            # Return state and any other value we might want
            return state, loss

        # Training loop
        losses = []
        for _ in tqdm(range(num_epochs)):
            epoch_loss = []
            for support, _ in data_loader:
                self.state, loss = _train_step(self.state, support)
                epoch_loss.append(loss)
            losses.append(np.array(epoch_loss).mean())
        self.model = self.model.bind(self.state.params)

        return losses


def main(
    seed: int = 0,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    M: int = 3,
    plot: bool = False,
):
    # setup random keys
    key = jax.random.PRNGKey(seed)
    key, input_key, init_key = jax.random.split(key, 3)

    dataset = SyntheticTask(seed=seed, M=M)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate
    )

    # initialize model
    input_init = jax.random.normal(input_key, (batch_size, 1, 1))
    model = MLP()
    params = model.init(init_key, input_init)

    # initialize optimizer and loss
    optimizer = optax.adam(learning_rate=learning_rate)
    loss = optax.squared_error

    # init model + train
    pretrain = PreTrain(model, params, optimizer, loss)
    losses = pretrain.train_model(dataloader)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.show()


if __name__ == "__main__":
    typer.run(main)
