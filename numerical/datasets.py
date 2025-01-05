import torch
import torch.nn as nn
from generate_numbers import generate_number_grid
from torch.utils.data import Dataset

class MetaModuloDataset(Dataset):
    def __init__(
        self,
        n_tasks=1000,
        n_samples_per_task=20,
        n_moduli=20,
        range_max=100,
        skip: int = 1,
        train: bool = True,
        sigma: float = 0.1,
        data: str = "image",
        model: str = "cnn",
        bit_width: int = 8,
    ):
        assert skip in [1, 2], "Skip must be either 1 or 2."
        assert data in ["image", "bits", "number"], "Data must be either image, bits, or number."
        self.n_tasks = n_tasks
        self.n_samples_per_task = n_samples_per_task
        self.n_moduli = n_moduli
        self.range_max = range_max
        self.tasks = []
        self.skip = skip
        self.train = train
        self.sigma = sigma
        self.data = data
        self.model = model
        self.bit_width = bit_width

        self._generate_tasks()

    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        X_s, _, y_s, X_q, _, y_q, _ = self.tasks[idx]
        return X_s, y_s, X_q, y_q

    def _generate_tasks(self):
        """Generate tasks based on the data type."""
        offset = 0 if self.train else (self.n_moduli if self.skip == 1 else 1)
        mul = 1 if self.skip == 1 else 2
        moduli = list(range(1 + offset, mul * self.n_moduli + 1 + offset, self.skip))
        
        if self.train and self.n_tasks != self.n_moduli:
            ms = torch.tensor([moduli[idx] for idx in torch.randint(0, len(moduli), (self.n_tasks,))]) 
        else:
            ms = moduli
        for m in ms:
            if self.data == "image":
                self.tasks.append(self._generate_image_task(m))
            elif self.data == "bits":
                self.tasks.append(self._generate_bits_task(m))
            elif self.data == "number":
                self.tasks.append(self._generate_number_task(m))


    def _generate_image_task(self, m):
        X_s = torch.randint(0, self.range_max, (self.n_samples_per_task, 1)).float()
        y_s = (X_s % m).float() + torch.normal(0, self.sigma, size=X_s.size())
        
        # Generate images for support set
        X_image_s = torch.zeros((self.n_samples_per_task, 1, 32, 32))
        for i, number in enumerate(X_s[:, 0].int().numpy()):
            grid_image = generate_number_grid(number).reshape(1, 32, 32)
            X_image_s[i] = torch.from_numpy(grid_image).contiguous()

        X_q = torch.arange(0, self.range_max).float().unsqueeze(1)
        y_q = (X_q % m).float()

        # Generate images for query set
        X_image_q = torch.zeros((self.range_max, 1, 32, 32))
        for i, number in enumerate(X_q[:, 0].int().numpy()):
            grid_image = generate_number_grid(number).reshape(1, 32, 32)
            X_image_q[i] = torch.from_numpy(grid_image).contiguous()

        if self.model == "mlp":
            X_image_s = X_image_s.reshape(self.n_samples_per_task, 32 * 32)
            X_image_q = X_image_q.reshape(self.range_max, 32 * 32)
        elif self.model in ["lstm", "transformer"]:
            X_image_s = self._image_to_patches(X_image_s)
            X_image_q = self._image_to_patches(X_image_q)
        
        return (X_image_s, X_s, y_s, X_image_q, X_q, y_q, m)

    def _generate_bits_task(self, m):
        X_s = torch.randint(0, self.range_max, (self.n_samples_per_task, 1)).float()
        y_s = (X_s % m).float() + torch.normal(0, self.sigma, size=X_s.size())

        X_q = torch.arange(0, self.range_max).float().unsqueeze(1)
        y_q = (X_q % m).float()

        X_bits_s = self._generate_bitstrings(X_s.int())
        X_bits_q = self._generate_bitstrings(X_q.int())

        if self.model == "mlp":
            X_bits_s = X_bits_s.squeeze()
            X_bits_q = X_bits_q.squeeze()
        return (X_bits_s, X_s, y_s, X_bits_q, X_q, y_q, m)

    def _generate_number_task(self, m):
        X_s = torch.randint(0, self.range_max, (self.n_samples_per_task, 1)).float()
        y_s = (X_s % m).float() + torch.normal(0, self.sigma, size=X_s.size())
        X_q = torch.arange(0, self.range_max).float().unsqueeze(1)
        y_q = (X_q % m).float()
        return (X_s, X_s, y_s, X_q, X_s, y_q, m)
    
    def _generate_bitstrings(self, numbers):
        binary_format = torch.arange(self.bit_width - 1, -1, -1, dtype=torch.int32)
        bitstrings = (numbers >> binary_format) & 1
        return bitstrings.unsqueeze(-1).float()

    def _image_to_patches(self, image_batch, patch_size=4):
        """Convert images into patches for models like transformers."""
        B, C, H, W = image_batch.shape
        assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"
        patches = image_batch.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.reshape(B, C, -1, patch_size, patch_size).permute(0, 2, 1, 3, 4)
        return patches.reshape(B, -1, C * patch_size * patch_size)