import torch
import torch.nn as nn
from generate_concepts import generate_concept
from grammer import DNFHypothesis
from torch.utils.data import Dataset

import numpy as np

feature_values = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 1, 1],
                      [0, 1, 0, 0],
                      [0, 1, 0, 1],
                      [0, 1, 1, 0],
                      [0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 1],
                      [1, 0, 1, 0],
                      [1, 0, 1, 1],
                      [1, 1, 0, 0],
                      [1, 1, 0, 1],
                      [1, 1, 1, 0],
                      [1, 1, 1, 1]])


class MetaBitConceptsDataset(Dataset):
    def __init__(
        self,
        n_tasks=10000,
        n_samples_per_task=9,
        data: str = "image",
        model: str = "cnn",
    ):
        assert n_samples_per_task < 15, "n_samples too high."
        assert data in ["image", "bits"], "Data must be either image or bits."
        self.n_tasks = n_tasks
        self.n_samples_per_task = n_samples_per_task
        self.tasks = []
        self.data = data
        self.model = model

        self._generate_tasks()

    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        X_s, _, y_s, X_q, _, y_q, _ = self.tasks[idx]
        return X_s, y_s, X_q, y_q
    
    def _generate_tasks(self):
        if self.data == "image":
            self._generate_image_tasks()
        elif self.data == "bits":
           self._generate_bit_tasks()

    def _generate_image_tasks(self):
        X_q = torch.tensor(feature_values, dtype=torch.float)
        X_image_q = torch.zeros((16, 3, 32, 32))
        for i, bits in enumerate(feature_values):
            grid_image = generate_concept(bits).reshape(3, 32, 32)
            X_image_q[i] = torch.from_numpy(grid_image).contiguous()
        if self.model == "mlp":
            X_image_q = X_image_q.reshape(16, 32 * 32 * 3)
        elif self.model in ["lstm", "transformer"]:
            X_image_q = self._image_to_patches(X_image_q)
        while len(self.tasks) < self.n_tasks:
            hyp = DNFHypothesis(n_features=4, no_true_false_top=True, b=1.0)
            labels = []
            for features in feature_values:
                labels.append(hyp.function(features))
            for i in range(len(labels)-1):
                if labels[i] != labels[i+1]:
                    inds = np.random.choice(16, size=(self.n_samples_per_task,))
                    X_s = X_q[inds]
                    X_image_s = X_image_q[inds]
                    y_s = torch.tensor(labels, dtype=torch.float).unsqueeze(1)[inds]
                    y_q = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
                    self.tasks.append((X_image_s, X_s, y_s, X_image_q, X_q, y_q, None))
                    break

    def _generate_bit_tasks(self):
        X_q = torch.tensor(feature_values, dtype=torch.float) # constant
        if self.model in ['lstm', 'transformer']:
            X_q = X_q.unsqueeze(2)
        while len(self.tasks) < self.n_tasks:
            hyp = DNFHypothesis(n_features=4, no_true_false_top=True, b=1.0)
            labels = []
            for features in feature_values:
                labels.append(hyp.function(features))
            for i in range(len(labels)-1):
                if labels[i] != labels[i+1]:
                    inds = np.random.choice(16, size=(self.n_samples_per_task,))
                    X_s = torch.tensor(feature_values[inds], dtype=torch.float)
                    y_s = torch.tensor(labels, dtype=torch.float).unsqueeze(1)[inds]
                    y_q = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
                    if self.model in ['lstm', 'transformer']:
                        X_s = X_s.unsqueeze(2)
                    self.tasks.append((X_s, X_s, y_s, X_q, X_q, y_q, None))
                    break

    def _image_to_patches(self, image_batch, patch_size=4):
        """Convert images into patches for models like transformers."""
        B, C, H, W = image_batch.shape
        assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"
        patches = image_batch.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.reshape(B, C, -1, patch_size, patch_size).permute(0, 2, 1, 3, 4)
        return patches.reshape(B, -1, C * patch_size * patch_size)

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    dataset = MetaBitConceptsDataset()
    print(len(dataset.tasks))
