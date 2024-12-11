import torch
import torch.nn as nn

from generate_numbers import generate_number_grid


class MetaNumberModuloDataset:
    def __init__(self, n_tasks=20, n_samples_per_task=100, range_max=100):
        self.n_tasks = n_tasks
        self.n_samples_per_task = n_samples_per_task
        self.range_max = range_max
        self.tasks = []

        # Generate tasks with different modulo values
        for m in range(1, n_tasks + 1):
            X_s = (
                torch.randint(0, range_max, size=(n_samples_per_task,))
                .float()
                .unsqueeze(1)
            )
            y_s = (X_s % m).float()
            X_q = torch.arange(0, range_max).float().unsqueeze(1)
            y_q = (X_q % m).float()
            self.tasks.append((X_s, y_s, X_q, y_q, m))

    def sample_task(self):
        idx = torch.randint(0, len(self.tasks), size=(1,)).item()
        return self.tasks[idx]


class MetaImageModuloDataset:
    def __init__(self, n_tasks=20, n_samples_per_task=100, range_max=100):
        self.n_tasks = n_tasks
        self.n_samples_per_task = n_samples_per_task
        self.range_max = range_max
        self.tasks = []

        # Generate tasks with different modulo values
        for m in range(1, n_tasks + 1):
            X_s = (
                torch.randint(0, range_max, size=(n_samples_per_task,))
                .float()
                .unsqueeze(1)
            )
            y_s = (X_s % m).float()

            X_image_s = torch.zeros((n_samples_per_task, 1, 32, 32))
            for i, number in enumerate(X_s[:, 0].int().numpy()):
                grid_image = generate_number_grid(number).reshape(1, 32, 32)
                X_image_s[i] = torch.tensor(grid_image)

            X_q = torch.arange(0, range_max).float().unsqueeze(1)
            y_q = (X_q % m).float()

            X_image_q = torch.zeros((range_max, 1, 32, 32))
            for i, number in enumerate(X_q[:, 0].int().numpy()):
                grid_image = generate_number_grid(number).reshape(1, 32, 32)
                X_image_q[i] = torch.tensor(grid_image)

            self.tasks.append((X_image_s, X_s, y_s, X_image_q, X_q, y_q, m))

    def sample_task(self):
        idx = torch.randint(0, len(self.tasks), size=(1,)).item()
        return self.tasks[idx]

class MetaBitStringModuloDataset:
    def __init__(self, n_tasks=20, n_samples_per_task=100, range_max=100, bit_width: int = 8):
        self.n_tasks = n_tasks
        self.n_samples_per_task = n_samples_per_task
        self.range_max = range_max
        self.tasks = []
        self.bit_width = bit_width

        # Generate tasks with different modulo values
        for m in range(1, n_tasks + 1):
            # Support set
            X_s = torch.randint(0, range_max, size=(n_samples_per_task, 1)).float()
            y_s = (X_s % m).float()

            # Query set
            X_q = torch.arange(0, range_max).float().unsqueeze(1)
            y_q = (X_q % m).float()

            # Generate bit representations for support and query sets
            X_bits_s = self.generate_bitstrings(X_s.int())
            X_bits_q = self.generate_bitstrings(X_q.int())

            self.tasks.append((X_bits_s, X_s, y_s, X_bits_q, X_q, y_q, m))

    def generate_bitstrings(self, numbers):
        binary_format = torch.arange(self.bit_width - 1, -1, -1, dtype=torch.int32)
        bitstrings = (numbers >> binary_format) & 1
        return bitstrings.unsqueeze(-1).float()

    def sample_task(self):
        idx = torch.randint(0, len(self.tasks), size=(1,)).item()
        return self.tasks[idx]