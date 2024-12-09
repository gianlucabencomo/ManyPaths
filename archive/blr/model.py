from flax import linen as nn

import jax
import jax.numpy as jnp


class MLP(nn.Module):
    n_hidden: int = 40
    n_layers: int = 2
    n_out: int = 1

    def setup(self):
        self.hidden_layers = [
            nn.Dense(features=self.n_hidden) for _ in range(self.n_layers)
        ]
        self.out = nn.Dense(features=self.n_out)

    def __call__(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = nn.relu(x)
        x = self.out(x)
        return x


class RNNCell(nn.Module):
    n_hidden: int = 40

    def setup(self):
        self.Whx = nn.Dense(features=self.n_hidden)

    def __call__(self, state, x):
        combined = jnp.concatenate([state, x], axis=-1)
        state = nn.tanh(self.Whx(combined))
        return state


class RNNModel(nn.Module):
    n_hidden: int = 128
    n_out: int = 1

    def setup(self):
        self.rnn_cell = RNNCell(self.n_hidden)
        self.out = nn.Dense(features=self.n_out)

    def __call__(self, inputs):
        # Initialize state to zero
        state = jnp.zeros((inputs.shape[0], self.n_hidden))
        outputs = []
        for t in range(inputs.shape[1]):
            state = self.rnn_cell(state, inputs[:, t, :])
            outputs.append(state)
        outputs = jnp.stack(outputs, axis=1)
        # Apply output transformation to each time step
        return jax.vmap(self.out)(outputs)
