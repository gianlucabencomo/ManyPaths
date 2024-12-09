import typer

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from basis import fourier_basis


def generate_synthetic(
    n_train: int = 100,
    n_test: int = 100,
    domain: tuple[float, float] = (-10, 10),
    basis: callable = fourier_basis,
    mu: float = 0.0,
    sigma: float = 1.0,
    M: int = 3,
    eps: float = 0.1,
    mixed: bool = True,
):
    X = np.sort(
        np.random.uniform(domain[0], domain[1], size=(n_train + n_test, 1)), axis=0
    )
    X, y, weights = sample_prior(X, basis=basis, mu=mu, sigma=sigma, M=M)

    train_inds = (
        np.sort(np.random.choice(n_train + n_test, size=(n_train), replace=False))
        if mixed
        else np.arange(0, n_train)
    )
    test_inds = np.delete(np.arange(0, n_train + n_test), train_inds)

    X_train, X_test = X[train_inds], X[test_inds]

    y_train = y[train_inds] + eps * np.random.normal(size=(n_train, 1))
    y_test = y[test_inds] + eps * np.random.normal(size=(n_test, 1))

    return X_train, X_test, y_train, y_test, weights


def sample_prior(
    X: np.array,
    basis: callable = fourier_basis,
    mu: float = 0.0,
    sigma: float = 1.0,
    M: int = 3,
):
    phi = basis(X, M=M)
    weights = np.random.normal(mu, sigma, size=(2 * M + 1, 1))
    y = phi @ weights
    return X, y, weights


class SyntheticTask(Dataset):
    def __init__(
        self,
        seed: int = 0,
        n_episodes: int = 10000,
        domain: tuple[float, float] = (-20, 20),
        basis: callable = fourier_basis,
        mu: float = 0.0,
        sigma: float = 1.0,
        M: int = 3,
        eps: float = 0.1,
        n_support: int = 50,
        n_query: int = 50,
        mixed: bool = True,
        sequential: bool = False,
    ):
        super().__init__()
        np.random.seed(seed)
        self.n_episodes = n_episodes
        self.domain = domain

        self.x_support = np.zeros((self.n_episodes, n_support, 1))
        self.y_support = np.zeros((self.n_episodes, n_support, 1))
        self.x_query = np.zeros((self.n_episodes, n_query, 1))
        self.y_query = np.zeros((self.n_episodes, n_query, 1))
        for i in range(n_episodes):
            (
                self.x_support[i],
                self.x_query[i],
                self.y_support[i],
                self.y_query[i],
                _,
            ) = generate_synthetic(
                n_support, n_query, domain, basis, mu, sigma, M, eps, mixed
            )

        if sequential:
            self.x_support = self.x_support[:, :-1, :]
            self.y_support = self.y_support[:, 1:, :]
            self.x_query = self.x_query[:, :-1, :]
            self.y_query = self.y_query[:, 1:, :]

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        support = (self.x_support[idx], self.y_support[idx])
        query = (self.x_query[idx], self.y_query[idx])
        return support, query


class SineWaveTask(Dataset):
    def __init__(
        self,
        seed: int = 0,
        n_episodes: int = 2000000,
        domain_support: tuple[float, float] = (-5.0, 5.0),
        domain_query: tuple[float, float] = (-5.0, 5.0),
        phase: tuple[float, float] = (0, np.pi),
        amplitude: tuple[float, float] = (0.1, 5.0),
        n_support: int = 5,
        n_query: int = 20,
        sequential: bool = False,
    ):
        super().__init__()
        np.random.seed(seed)
        self.n_episodes = n_episodes
        self.domain_support = domain_support
        self.domain_query = domain_query
        self.phase = np.random.uniform(
            low=phase[0], high=phase[1], size=(self.n_episodes, 1, 1)
        )
        self.amplitude = np.random.uniform(
            low=amplitude[0], high=amplitude[1], size=(self.n_episodes, 1, 1)
        )

        self.x_support = np.sort(
            np.random.uniform(
                low=self.domain_support[0],
                high=self.domain_support[1],
                size=(self.n_episodes, n_support, 1),
            ),
            axis=1,
        )
        self.y_support = self.sine(self.x_support, a=self.amplitude, b=self.phase)

        self.x_query = np.sort(
            np.random.uniform(
                low=self.domain_query[0],
                high=self.domain_query[1],
                size=(self.n_episodes, n_query, 1),
            ),
            axis=1,
        )
        self.y_query = self.sine(self.x_query, a=self.amplitude, b=self.phase)

        if sequential:
            self.x_support = self.x_support[:, :-1, :]
            self.y_support = self.y_support[:, 1:, :]
            self.x_query = self.x_query[:, :-1, :]
            self.y_query = self.y_query[:, 1:, :]

    def sine(self, x, a: float = 1.0, b: float = 1.0, c: float = 0.0, d: float = 0.0):
        return a * np.sin(b * (x + c)) + d

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        support = (self.x_support[idx], self.y_support[idx])
        query = (self.x_query[idx], self.y_query[idx])
        return support, query


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def plot(seed: int = 0, basis: str = "fourier", M: int = 3, test: bool = False):
    np.random.seed(seed)
    if basis != "fourier":
        raise ValueError("Basis function not recognized.")
    else:
        basis_function = fourier_basis

    # unit test
    if test:
        dataset = SyntheticTask(seed=seed)
        dataloader = DataLoader(
            dataset, batch_size=32, shuffle=True, collate_fn=numpy_collate
        )
        for support, query in iter(dataloader):
            continue

    plt.figure(figsize=(16, 8))  # Adjusted for subplots

    for i in range(16):
        plt.subplot(4, 4, i + 1)  # Set subplot position
        domain = (-20, 20)
        X_train, X_test, y_train, y_test, weights = generate_synthetic(
            basis=basis_function, domain=domain, M=M
        )

        # Scatter plots for training and testing data
        plt.scatter(X_train, y_train, color="blue", label="Training Set", alpha=0.5)
        plt.scatter(X_test, y_test, color="red", label="Test Set", alpha=0.5)

        # Optionally, plot the actual basis-generated curve for reference
        x = np.linspace(domain[0], domain[1], 500).reshape(500, 1)
        y = basis_function(x, M=M) @ weights
        plt.plot(x, y, color="green", label="True Fourier Series")

        plt.title(f"Sample {i+1}")
        plt.xlabel("X")
        plt.ylabel("y")

        if i == 0:
            plt.legend()  # Show legend only on the first subplot to save space

        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    typer.run(plot)
