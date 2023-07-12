import numpy as np
import jax.numpy as jnp


def sample_tasks(outer_size, inner_size):
    """Generates samples from the original MAML Sine Wave Task.

    Taken directly from:
    https://blog.evjang.com/2019/02/maml-jax.html

    """
    As = []
    phases = []
    for _ in range(outer_size):
        As.append(np.random.uniform(low=0.1, high=0.5))
        phases.append(np.random.uniform(low=0.0, high=np.pi))

    def get_batch():
        xs, ys = [], []
        for A, phase in zip(As, phases):
            x = np.random.uniform(low=-5.0, high=5.0, size=(inner_size, 1))
            y = A * np.sin(x + phase)
            xs.append(x)
            ys.append(y)
        return jnp.stack(xs), jnp.stack(ys)

    x1, y1 = get_batch()
    x2, y2 = get_batch()
    return x1, y1, x2, y2
