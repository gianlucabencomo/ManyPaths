import typer
import numpy as np
from numpy.linalg import inv, det, eigvals

import matplotlib.pyplot as plt

from data import generate_synthetic
from basis import identity_basis, polynomial_basis, gaussian_basis, fourier_basis


def plot_posterior_predictive(xgrid, mu, sigma, X, y_true=None, tt_split=None):
    """
    Plot the posterior predictive distribution along with the original data points.

    Args:
        X (np.array): The input features.
        mu (np.array): Mean of the posterior predictive distribution.
        sigma (np.array): Variance of the posterior predictive distribution.
        y_true (np.array, optional): True target values. If provided, they will be plotted for reference.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(xgrid, mu, label="Predictive mean")
    plt.fill_between(
        xgrid.flatten(),
        (mu - 1.96 * np.sqrt(sigma)).flatten(),
        (mu + 1.96 * np.sqrt(sigma)).flatten(),
        color="lightblue",
        alpha=0.5,
        label="95% confidence interval",
    )

    if y_true is not None and not tt_split:
        plt.scatter(X, y_true, color="red", alpha=0.5, label="Ground Truth")
    elif y_true is not None and tt_split:
        plt.scatter(
            X[:tt_split],
            y_true[:tt_split],
            color="red",
            alpha=0.5,
            label="Training Ground Truth",
        )
        plt.scatter(
            X[tt_split:],
            y_true[tt_split:],
            color="blue",
            alpha=0.5,
            label="Test Ground Truth",
        )

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Posterior Predictive Distribution")
    plt.show()


class BayesianLinearRegressor:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def initialize(self, alpha: float, beta: float) -> tuple[np.array, np.array]:
        self.mean = np.zeros((self.dim, 1))
        self.precision = alpha * np.eye(self.dim)
        self.alpha = alpha
        self.beta = beta
        return self.mean, self.precision

    def log_posterior(self, phi: np.array, t: np.array) -> float:
        log_ll = -self.beta * 0.5 * np.sum((t - phi @ self.mean) ** 2)
        log_prior = -self.alpha * 0.5 * self.mean.T @ self.mean
        return (log_ll + log_prior).flatten()[0]

    def update(self, phi: np.array, t: np.array) -> np.array:
        precision = self.precision + self.beta * phi.T @ phi
        self.mean = inv(precision) @ (
            self.precision @ self.mean + self.beta * phi.T @ t
        )
        self.precision = precision
        return self.mean, self.precision

    def posterior_predictive(self, phi: np.array) -> tuple[float, float]:
        mu = phi @ self.mean
        # Only compute variances (diagonal elements of covariance matrix)
        sigma = 1.0 / self.beta + np.sum(phi.dot(inv(self.precision)) * phi, axis=1)
        return mu, sigma.reshape(-1, 1)

    def log_marginal_likelihood(self, phi: np.array, t: np.array) -> float:
        M = self.mean.shape[0]
        N = phi.shape[0]
        A = self.alpha * np.eye(M) + self.beta * phi.T @ phi
        E_m = (
            0.5 * self.beta * np.sum((t - phi @ self.mean) ** 2)
            + 0.5 * self.alpha * self.mean.T @ self.mean
        )
        return (
            (M / 2) * np.log(self.alpha)
            + (N / 2) * np.log(self.beta)
            - E_m
            - 0.5 * np.log(det(A))
            - (N / 2) * np.log(2 * np.pi)
        )

    def fit(
        self, phi: np.array, t: np.array, max_iter: int = 100, tol: float = 1e-8
    ) -> None:
        N = phi.shape[0]

        for i in range(max_iter):
            self.update(phi, t)
            lam = eigvals(self.beta * phi.T @ phi)
            gamma = np.sum(lam / (self.alpha + lam))
            self.alpha = gamma / (self.mean.T @ self.mean)
            self.beta = 1 / ((1 / (N - gamma)) * np.sum((t - phi @ self.mean) ** 2))


def main(
    seed: int = 0,
    basis: str = "fourier",
    M: int = 3,
    domain: float = 20.0,
    alpha: float = 1.0,
    beta: float = 10.0,
    fit: bool = False,
    plot: bool = False,
    verbose: bool = False,
):
    # set random seed
    np.random.seed(seed)
    # basis function
    if basis != "fourier":
        raise ValueError("Basis function not recognized.")
    else:
        basis_function = fourier_basis
    # domain
    domain = (-domain, domain)

    # load data
    X_train, X_test, y_train, y_test, weights = generate_synthetic(
        basis=basis_function, domain=domain, M=M, mixed=False
    )

    # apply basis functions to train and test data
    phi_train = fourier_basis(X_train, M=M)
    phi_test = fourier_basis(X_test, M=M)

    # initialize BLR
    N, D = phi_train.shape
    blr = BayesianLinearRegressor(dim=D)
    blr.initialize(alpha=np.array([[alpha]]), beta=beta)

    # update model
    if verbose and fit:
        print(
            f"Log posterior before update = {blr.log_posterior(phi_test, y_test):.3f}"
        )

    # optimize model via empirical Bayes
    if fit:
        blr.fit(phi_train, y_train)
    if verbose:
        print(
            f"Hyperparameters : alpha = {blr.alpha[0][0]:.3f}, beta = {blr.beta:.3f} {('(Optimized)' if fit else '')}"
        )

    blr.update(phi_train, y_train)
    if verbose:
        print(
            f"Log posterior {('after update ' if fit else '')}= {blr.log_posterior(phi_test, y_test):.3f}"
        )

    # calculate posterior predictive distribution for test data
    mu, sigma = blr.posterior_predictive(phi_test)

    if plot:
        xgrid = np.linspace(domain[0], domain[1], 10000).reshape(-1, 1)
        phi = basis_function(xgrid, M=M)
        mu, sigma = blr.posterior_predictive(phi)
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        plot_posterior_predictive(
            xgrid, mu, sigma, X, y_true=y, tt_split=X_train.shape[0]
        )


if __name__ == "__main__":
    typer.run(main)
