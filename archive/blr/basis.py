import typer
import numpy as np

import matplotlib.pyplot as plt


def identity_basis(X: np.array) -> np.array:
    """Generates an identity basis (design) matrix with an added bias term.

    Args:
        X (np.array): The input dataset of shape (N, D) where N is the number of samples
                      and D is the number of features.

    Returns:
        np.array: The design matrix of shape (N, D + 1) which includes the original features
                  and an added bias term as the first column.
    """
    N, D = X.shape
    phi = np.concatenate((np.ones((N, 1)), X), axis=1)  # N x (D + 1) design matrix
    return phi


def polynomial_basis(X: np.array, M: int = 3) -> np.array:
    """Generates a polynomial basis (design) matrix of degree M.

    Args:
        X (np.array): The input dataset of shape (N, D) where N is the number of samples
                      and D is the number of features.
        M (int): The degree of the polynomial basis.

    Returns:
        np.array: The design matrix of shape (N, 1 + D * M) which includes the polynomial
                  terms of the original features up to degree M and a bias term.
    """
    N, D = X.shape
    phi = np.ones((N, 1 + D * M))
    for i in range(0, M):
        for j in range(D):
            phi[:, i * D + j + 1] = X[:, j] ** (i + 1)
    return phi


def polynomial_interaction_basis(X: np.array, degree: int = 2) -> np.array:
    """
    Generates a polynomial basis matrix of degree 2 with interaction terms.

    Args:
        X (np.array): The input dataset of shape (N, D) where N is the number of samples
                      and D is the number of features.
        degree (int): The degree of the polynomial basis. Defaults to 2. Currently,
                      the implementation specifically handles degree 2 with interactions.

    Returns:
        np.array: The design matrix which includes the polynomial terms up to degree 2
                  and interaction terms, along with a bias term.
    """
    if degree != 2:
        raise ValueError("This function is specifically implemented for degree 2.")

    N, D = X.shape
    # For degree 2 with interaction terms, the number of terms is D (linear) + D (squared) + D*(D-1)/2 (interactions) + 1 (bias)
    num_terms = D + D + (D * (D - 1)) // 2 + 1
    phi = np.ones((N, num_terms))

    idx = 1  # Start after bias term
    # Linear and squared terms
    for j in range(D):
        phi[:, idx] = X[:, j]  # Linear term
        idx += 1
        phi[:, idx] = X[:, j] ** 2  # Squared term
        idx += 1
    # Interaction terms
    for j in range(D):
        for k in range(j + 1, D):
            phi[:, idx] = X[:, j] * X[:, k]
            idx += 1

    return phi


def gaussian_basis(
    X: np.array, M: int = 3, mu: np.array = None, s: float = 1.0
) -> np.array:
    """Generates a Gaussian basis (design) matrix.

    Args:
        X (np.array): The input dataset of shape (N, D) where N is the number of samples
                      and D is the number of features.
        M (int): The number of Gaussian functions to use.
        mu (np.array, optional): The means of the Gaussian functions. If None, the means
                                 are linearly spaced between the min and max of the dataset.
                                 Defaults to None.
        s (float, optional): The standard deviation of the Gaussian functions. Defaults to 1.0.

    Returns:
        np.array: The design matrix of shape (N, 1 + M * D) which includes the Gaussian
                  features of the original features and a bias term.
    """
    N, D = X.shape
    if mu is None:
        mu = np.linspace(X.min(), X.max(), M)
    phi = np.ones((N, 1 + M * D))
    for i in range(M):
        for j in range(D):
            phi[:, i * D + j + 1] = np.exp(-0.5 * (X[:, j] - mu[i]) ** 2 / s**2)
    return phi


def fourier_basis(X: np.array, M: int = 3) -> np.array:
    """
    Generates a Fourier basis (design) matrix.

    Args:
        X (np.array): The input dataset of shape (N,) where N is the number of samples.
                      X should be a one-dimensional array for this implementation.
        M (int): The number of frequency components (pairs of sine and cosine) to use.

    Returns:
        np.array: The design matrix of shape (N, 1 + 2 * M) which includes a bias term,
                  and sine and cosine terms for M frequencies.
    """
    N = X.shape[0]
    # Ensure X is a one-dimensional array
    if X.shape[1] != 1:
        raise ValueError("Input array X must be one-dimensional.")

    X = X.flatten()

    # Initialize the design matrix with the bias term
    phi = np.ones((N, 1 + 2 * M))

    # Generate sine and cosine terms
    for m in range(1, M + 1):
        phi[:, 2 * m - 1] = np.cos(m * X)
        phi[:, 2 * m] = np.sin(m * X)
    return phi


def visualize_basis_functions(X, y, M=3):
    """Visualizes the original data and its transformation using identity,
    polynomial, and Gaussian basis functions.

    Args:
        X (np.array): The input dataset of shape (N, 1) where N is the number of samples.
        y (np.array): The target values corresponding to X.
        M (int): The parameter for the polynomial and Gaussian basis functions
                 indicating the degree or the number of Gaussians.
    """
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))

    # Identity Basis
    phi_identity = identity_basis(X)
    axs[0].scatter(X, y, color="blue", label="Original Data")
    axs[0].set_title("Identity Basis")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("y")
    axs[0].legend()

    # Polynomial Basis
    phi_polynomial = polynomial_basis(X, M=M)
    for i in range(1, M + 1):
        axs[1].scatter(X, phi_polynomial[:, i], label=f"x^{i}")
    axs[1].set_title("Polynomial Basis")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Transformed Features")
    axs[1].legend()

    # Gaussian Basis
    phi_gaussian = gaussian_basis(X, M=M)
    for i in range(1, M + 1):
        axs[2].scatter(X, phi_gaussian[:, i], label=f"Gaussian {i}")
    axs[2].set_title("Gaussian Basis")
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Transformed Features")
    axs[2].legend()

    # Gaussian Basis
    phi_fourier = fourier_basis(X, M=M)
    for i in range(1, M + 1):
        axs[3].scatter(X, phi_fourier[:, 2 * i - 1], label=f"Fourier {i} Component 1")
        axs[3].scatter(X, phi_fourier[:, 2 * i], label=f"Fourier {i} Component 2")
    axs[3].set_title("Fourier Basis")
    axs[3].set_xlabel("X")
    axs[3].set_ylabel("Transformed Features")
    axs[3].legend()

    # Original Data for comparison
    axs[4].scatter(X, y, color="red", label="Original Data")
    axs[4].set_title("Original Data")
    axs[4].set_xlabel("X")
    axs[4].set_ylabel("y")
    axs[4].legend()

    plt.tight_layout()
    plt.show()


def main(seed: int = 0, N: int = 5, D: int = 1, M: int = 5) -> None:
    """Runs basic tests on the basis functions with randomly generated data.

    Args:
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        N (int, optional): Number of samples in the randomly generated dataset. Defaults to 5.
        D (int, optional): Number of features in the randomly generated dataset. Defaults to 1.
        M (int, optional): Degree for polynomial basis and number of Gaussians for Gaussian basis.
                           Defaults to 5.
    """
    np.random.seed(seed)  # Set seed for reproducibility
    X = np.random.normal(size=(N, D))

    # Run some basic tests to verify the shape of the output matrices
    assert identity_basis(X).shape == (N, D + 1)
    assert polynomial_basis(X, M=M).shape == (N, M * D + 1)
    assert gaussian_basis(X, M=M).shape == (N, M * D + 1)
    assert polynomial_interaction_basis(X).shape == (N, D + D + (D * (D - 1)) // 2 + 1)

    # visualize this basis functions with a sine wave
    N = 100  # Number of samples
    X = np.linspace(-5, 5, N).reshape(-1, 1)  # Input data
    y = np.sin(X)  # Target values

    visualize_basis_functions(X, y, M=3)


if __name__ == "__main__":
    typer.run(main)
