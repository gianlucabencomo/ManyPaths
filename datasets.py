import torch
import numpy as np
from torch.utils.data import Dataset
from generate_numbers import generate_number_grid
from generate_concepts import generate_concept
from grammer import DNFHypothesis
from constants import FEATURE_VALUES

class BaseMetaDataset(Dataset):
    def __init__(self):
        self.tasks = []

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        X_s, _, y_s, X_q, _, y_q, _ = self.tasks[idx]
        return X_s, y_s, X_q, y_q

    def _image_to_patches(self, image_batch, patch_size=4):
        B, C, H, W = image_batch.shape
        assert H % patch_size == 0 and W % patch_size == 0
        patches = image_batch.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.reshape(B, C, -1, patch_size, patch_size).permute(0, 2, 1, 3, 4)
        return patches.reshape(B, -1, C * patch_size * patch_size)

class MetaBitConceptsDataset(BaseMetaDataset):
    def __init__(self, n_tasks=10000, n_samples_per_task=20, data="image", model="cnn"):
        super().__init__()
        assert data in ["image", "bits"]
        self.n_tasks = n_tasks
        self.n_samples_per_task = n_samples_per_task
        self.data = data
        self.model = model
        self._generate_tasks()
    
    def _generate_tasks(self):
        if self.data == "image":
            self._generate_image_tasks()
        else:
           self._generate_bit_tasks()

    def _generate_image_tasks(self):
        X_q = torch.tensor(FEATURE_VALUES, dtype=torch.float)
        X_image_q = torch.zeros((16, 3, 32, 32))
        for i, bits in enumerate(FEATURE_VALUES):
            grid_image = generate_concept(bits, scale=255.0).reshape(3, 32, 32)
            X_image_q[i] = torch.from_numpy(grid_image)
        mean = X_image_q.mean(dim=[0, 2, 3])
        std = X_image_q.std(dim=[0, 2, 3])
        X_image_q = (X_image_q - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        if self.model == "mlp":
            X_image_q = X_image_q.reshape(16, 32 * 32 * 3)
        elif self.model in ["lstm", "transformer"]:
            X_image_q = self._image_to_patches(X_image_q)
        while len(self.tasks) < self.n_tasks:
            hyp = DNFHypothesis(n_features=4, no_true_false_top=True, b=1.0)
            labels = [hyp.function(f) for f in FEATURE_VALUES]
            for i in range(len(labels) - 1):
                if labels[i] != labels[i + 1]:
                    m = np.random.randint(1, high=self.n_samples_per_task)
                    inds = np.random.choice(16, size=(m,))
                    X_s = X_q[inds]
                    X_image_s = X_image_q[inds]
                    y_s = torch.tensor(labels, dtype=torch.float).unsqueeze(1)[inds]
                    y_q = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
                    self.tasks.append((X_image_s, X_s, y_s, X_image_q, X_q, y_q, m))
                    break

    def _generate_bit_tasks(self):
        X_q = torch.tensor(FEATURE_VALUES, dtype=torch.float)
        if self.model in ['lstm', 'transformer']:
            X_q = X_q.unsqueeze(2)
        while len(self.tasks) < self.n_tasks:
            hyp = DNFHypothesis(n_features=4, no_true_false_top=True, b=1.0)
            labels = [hyp.function(f) for f in FEATURE_VALUES]
            for i in range(len(labels) - 1):
                if labels[i] != labels[i + 1]:
                    m = np.random.randint(1, high=self.n_samples_per_task)
                    inds = np.random.choice(16, size=(m,))
                    X_s = torch.tensor(FEATURE_VALUES[inds], dtype=torch.float)
                    y_s = torch.tensor(labels, dtype=torch.float).unsqueeze(1)[inds]
                    y_q = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
                    if self.model in ['lstm', 'transformer']:
                        X_s = X_s.unsqueeze(2)
                    X_bits_s = (X_s * 2 - 1)
                    X_bits_q = (X_q * 2 - 1)
                    self.tasks.append((X_bits_s, X_s, y_s, X_bits_q, X_q, y_q, m))
                    break

class MetaModuloDataset(BaseMetaDataset):
    def __init__(
        self,
        n_tasks=1000,
        n_samples_per_task=20,
        n_moduli=20,
        range_max=100,
        skip=1,
        train=True,
        sigma=0.1,
        data="image",
        model="cnn",
        bit_width=8,
    ):
        super().__init__()
        assert skip in [1, 2]
        assert data in ["image", "bits", "number"]
        self.n_tasks = n_tasks
        self.n_samples_per_task = n_samples_per_task
        self.n_moduli = n_moduli
        self.range_max = range_max
        self.skip = skip
        self.train = train
        self.sigma = sigma
        self.data = data
        self.model = model
        self.bit_width = bit_width
        self._generate_tasks()

    def _generate_tasks(self):
        offset = 0 if self.train else (self.n_moduli if self.skip == 1 else 1)
        mul = 1 if self.skip == 1 else 2
        moduli = list(range(1 + offset, mul * self.n_moduli + 1 + offset, self.skip))
        if self.train and self.n_tasks != self.n_moduli:
            ms = torch.tensor([moduli[i] for i in torch.randint(0, len(moduli), (self.n_tasks,))])
        else:
            ms = moduli
        for m in ms:
            if self.data == "image":
                self.tasks.append(self._generate_image_task(m))
            elif self.data == "bits":
                self.tasks.append(self._generate_bits_task(m))
            else:
                self.tasks.append(self._generate_number_task(m))


    def _generate_image_task(self, m):
        X_s = torch.randint(0, self.range_max, (self.n_samples_per_task, 1)).float()
        y_s = (X_s % m).float() + torch.normal(0, self.sigma, size=X_s.size())
        X_q = torch.arange(0, self.range_max).float().unsqueeze(1)
        y_q = (X_q % m).float()
        X_image_q = torch.zeros((self.range_max, 1, 32, 32))
        for i, num in enumerate(X_q[:, 0].int().numpy()):
            grid_image = generate_number_grid(num).reshape(1, 32, 32)
            X_image_q[i] = torch.from_numpy(grid_image)
        mean = X_image_q.mean(dim=[0, 2, 3])
        std = X_image_q.std(dim=[0, 2, 3])
        X_image_q = (X_image_q - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        X_image_s = X_image_q[X_s[:, 0].int()]
        if self.model == "mlp":
            X_image_s = X_image_s.view(self.n_samples_per_task, 32 * 32)
            X_image_q = X_image_q.view(self.range_max, 32 * 32)
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
        b = torch.arange(self.bit_width - 1, -1, -1, dtype=torch.int32)
        bitstrings = (numbers >> b) & 1
        return (bitstrings * 2 - 1).unsqueeze(-1).float()