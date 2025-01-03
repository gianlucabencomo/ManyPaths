import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from grammer import DNFHypothesis

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

class MetaConceptDataset(ABC):
    def __init__(
        self,
        n_tasks=10000,
        n_samples_per_task=5,
    ):
        self.n_tasks = n_tasks
        self.n_samples_per_task = n_samples_per_task
        self.tasks = []
        self._generate_tasks()

    @abstractmethod
    def _generate_tasks(self):
        pass

    def sample_task(self):
        task = torch.randint(0, len(self.tasks), size=(1,)).item()
        return self.tasks[task]


class MetaBitConceptsDataset(MetaConceptDataset):
    def _generate_tasks(self):
        X_q = torch.tensor(feature_values, dtype=torch.float) # constant
        while len(self.tasks) < self.n_tasks:
            hyp = DNFHypothesis(n_features=4, no_true_false_top=True, b=1.0)
            labels = []
            for features in feature_values:
                labels.append(hyp.function(features))
            for i in range(len(labels)-1):
                if labels[i] != labels[i+1]:
                    inds = np.random.choice(16, size=(self.n_samples_per_task,))
                    X_s = torch.tensor(feature_values[inds], dtype=torch.float)
                    y_s = torch.tensor(labels)
                    y_q = torch.tensor(labels)
                    self.tasks.append((X_s, X_s, y_s, X_q, X_q, y_q, None))

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    dataset = MetaBitConceptsDataset()
    print(len(dataset.tasks))
