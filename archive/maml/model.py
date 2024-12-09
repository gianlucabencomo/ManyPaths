from flax import linen as nn  # Linen API


class MLP(nn.Module):
    """Basic MLP model."""

    hidden_dim: int = 40
    layers: int = 2
    out_dim: int = 1

    @nn.compact
    def __call__(self, x):
        for _ in range(self.layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x
