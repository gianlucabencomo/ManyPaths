import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from generate_numbers import generate_number_grid


class MetaModuloDataset(ABC):
    def __init__(
        self,
        n_tasks=20,
        n_samples_per_task=100,
        range_max=100,
        n_samples=1000, # how many random datapoints per task
        skip: int = 1,
        train: bool = True,
        sigma: float = 0.1,
        model: str = "cnn",
    ):
        assert skip in [1, 2], "Skip must be either 1 or 2."
        self.n_tasks = n_tasks
        self.n_samples_per_task = n_samples_per_task
        self.n_samples = n_samples
        self.range_max = range_max
        self.tasks = []
        self.skip = skip
        self.train = train
        self.sigma = sigma
        self.model = model

        self._generate_tasks()

    @abstractmethod
    def _generate_tasks(self):
        pass

    def sample_task(self):
        task = torch.randint(0, len(self.tasks), size=(1,)).item()
        idx = torch.randint(0, len(self.tasks[task]), size=(1,)).item()
        return self.tasks[task][idx]


class MetaNumberModuloDataset(MetaModuloDataset):
    def _generate_tasks(self):
        if self.model != "mlp":
            raise ValueError("Model must be 'mlp'")
        offset = 0 if self.train else (self.n_tasks if self.skip == 1 else 1)
        mul = 1 if self.skip == 1 else 2
        for m in range(1 + offset, mul * self.n_tasks + 1 + offset, self.skip):
            task = []
            for _ in range(self.n_samples):
                X_s = (
                    torch.randint(
                        0, self.range_max, size=(self.n_samples_per_task, 1)
                    ).float()
                    if self.n_samples_per_task != 100
                    else torch.arange(0, self.range_max).float().unsqueeze(1)
                )
                y_s = (X_s % m).float() + torch.normal(
                    mean=0.0, std=self.sigma, size=X_s.size()
                )
                X_q = torch.arange(0, self.range_max).float().unsqueeze(1)
                y_q = (X_q % m).float()
                task.append((X_s, X_s, y_s, X_q, X_q, y_q, m))
            self.tasks.append(task)


class MetaImageModuloDataset(MetaModuloDataset):
    def _generate_tasks(self):
        offset = 0 if self.train else (self.n_tasks if self.skip == 1 else 1)
        mul = 1 if self.skip == 1 else 2
        for m in range(1 + offset, mul * self.n_tasks + 1 + offset, self.skip):
            task = []
            for _ in range(self.n_samples):
                X_s = (
                    torch.randint(
                        0, self.range_max, size=(self.n_samples_per_task, 1)
                    ).float()
                    if self.n_samples_per_task != 100
                    else torch.arange(0, self.range_max).float().unsqueeze(1)
                )
                y_s = (X_s % m).float() + torch.normal(
                    mean=0.0, std=self.sigma, size=X_s.size()
                )

                X_image_s = torch.zeros((self.n_samples_per_task, 1, 32, 32))
                for i, number in enumerate(X_s[:, 0].int().numpy()):
                    grid_image = generate_number_grid(number).reshape(1, 32, 32)
                    X_image_s[i] = torch.tensor(grid_image)

                X_q = torch.arange(0, self.range_max).float().unsqueeze(1)
                y_q = (X_q % m).float()

                X_image_q = torch.zeros((self.range_max, 1, 32, 32))
                for i, number in enumerate(X_q[:, 0].int().numpy()):
                    grid_image = generate_number_grid(number).reshape(1, 32, 32)
                    X_image_q[i] = torch.tensor(grid_image)

                if self.model == "mlp":
                    X_image_s = X_image_s.reshape(self.n_samples_per_task, 32 * 32)
                    X_image_q = X_image_q.reshape(self.range_max, 32 * 32)
                elif self.model in ["lstm", "transformer"]:
                    X_image_s = self.image_to_patches(X_image_s)
                    X_image_q = self.image_to_patches(X_image_q)
                task.append((X_image_s, X_s, y_s, X_image_q, X_q, y_q, m))
            self.tasks.append(task)

    def image_to_patches(self, image_batch, patch_size=4):
        B, C, H, W = image_batch.shape  # Batch size, channels, height, width
        assert (
            H % patch_size == 0 and W % patch_size == 0
        ), "Image dimensions must be divisible by patch size"
        patches = image_batch.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        patches = patches.reshape(
            B, C, -1, patch_size, patch_size
        )  # Flatten spatial dimensions
        patches = patches.permute(0, 2, 1, 3, 4)  # Move patch index to second dimension
        patches = patches.reshape(B, -1, patch_size * patch_size)  # Flatten each patch
        return patches


class MetaBitStringModuloDataset(MetaModuloDataset):
    def __init__(
        self,
        n_tasks=20,
        n_samples_per_task=100,
        range_max=100,
        n_samples=1000,
        bit_width: int = 8,
        skip: int = 1,
        train: bool = True,
        sigma: float = 0.1,
        model: str = "lstm",
    ):
        self.bit_width = bit_width
        super().__init__(
            n_tasks, n_samples_per_task, range_max, n_samples, skip, train, sigma, model
        )

    def _generate_tasks(self):
        if self.model not in ["lstm", "mlp", "transformer"]:
            raise ValueError("Model must be 'cnn', 'mlp', or 'transformer'")
        offset = 0 if self.train else (self.n_tasks if self.skip == 1 else 1)
        mul = 1 if self.skip == 1 else 2
        for m in range(1 + offset, mul * self.n_tasks + 1 + offset, self.skip):
            task = []
            for _ in range(self.n_samples):
                X_s = (
                    torch.randint(
                        0, self.range_max, size=(self.n_samples_per_task, 1)
                    ).float()
                    if self.n_samples_per_task != 100
                    else torch.arange(0, self.range_max).float().unsqueeze(1)
                )
                y_s = (X_s % m).float() + torch.normal(
                    mean=0.0, std=self.sigma, size=X_s.size()
                )

                X_q = torch.arange(0, self.range_max).float().unsqueeze(1)
                y_q = (X_q % m).float()

                X_bits_s = self.generate_bitstrings(X_s.int())
                X_bits_q = self.generate_bitstrings(X_q.int())

                if self.model == "mlp":
                    X_bits_s = X_bits_s.squeeze()
                    X_bits_q = X_bits_q.squeeze()
                task.append((X_bits_s, X_s, y_s, X_bits_q, X_q, y_q, m))
            self.tasks.append(task)

    def generate_bitstrings(self, numbers):
        binary_format = torch.arange(self.bit_width - 1, -1, -1, dtype=torch.int32)
        bitstrings = (numbers >> binary_format) & 1
        return bitstrings.unsqueeze(-1).float()
