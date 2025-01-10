import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms, datasets
from torchvision.datasets.utils import list_files
from collections import defaultdict
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
        patches = image_batch.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        patches = patches.reshape(B, C, -1, patch_size, patch_size).permute(
            0, 2, 1, 3, 4
        )
        return patches.reshape(B, -1, C * patch_size * patch_size)


class MetaBitConceptsDataset(BaseMetaDataset):
    def __init__(
        self,
        n_tasks: int = 10000,
        data: str = "image",
        model: str = "cnn",
        n_support: int = None,  # for test-time, testing across n_support #
    ):
        super().__init__()
        assert data in ["image", "bits"]
        self.n_tasks = n_tasks
        self.data = data
        self.model = model
        self.n_support = n_support
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
                    n_support = (
                        np.random.randint(2, high=20)
                        if self.n_support is None
                        else self.n_support
                    )
                    inds = np.random.choice(16, size=(n_support,))
                    X_s = X_q[inds]
                    X_image_s = X_image_q[inds]
                    y_s = torch.tensor(labels, dtype=torch.float).unsqueeze(1)[inds]
                    y_q = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
                    self.tasks.append((X_image_s, X_s, y_s, X_image_q, X_q, y_q, n_support))
                    break

    def _generate_bit_tasks(self):
        X_q = torch.tensor(FEATURE_VALUES, dtype=torch.float)
        if self.model in ["lstm", "transformer"]:
            X_q = X_q.unsqueeze(2)
        while len(self.tasks) < self.n_tasks:
            hyp = DNFHypothesis(n_features=4, no_true_false_top=True, b=1.0)
            labels = [hyp.function(f) for f in FEATURE_VALUES]
            for i in range(len(labels) - 1):
                if labels[i] != labels[i + 1]:
                    n_support = (
                        np.random.randint(2, high=20)
                        if self.n_support is None
                        else self.n_support
                    )
                    inds = np.random.choice(16, size=(n_support,))
                    X_s = torch.tensor(FEATURE_VALUES[inds], dtype=torch.float)
                    y_s = torch.tensor(labels, dtype=torch.float).unsqueeze(1)[inds]
                    y_q = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
                    if self.model in ["lstm", "transformer"]:
                        X_s = X_s.unsqueeze(2)
                    X_bits_s = X_s * 2 - 1
                    X_bits_q = X_q * 2 - 1
                    self.tasks.append((X_bits_s, X_s, y_s, X_bits_q, X_q, y_q, n_support))
                    break


class MetaModuloDataset(BaseMetaDataset):
    def __init__(
        self,
        n_tasks=10000,
        n_moduli=20,
        range_max=100,
        skip=1,
        train=True,
        sigma=0.1,
        data="image",
        model="cnn",
        bit_width=8,
        n_support: int = None,  # for test-time, testing across n_support #
    ):
        super().__init__()
        assert skip in [1, 2]
        assert data in ["image", "bits", "number"]
        self.n_tasks = n_tasks
        self.n_moduli = n_moduli
        self.range_max = range_max
        self.skip = skip
        self.train = train
        self.sigma = sigma
        self.data = data
        self.model = model
        self.bit_width = bit_width
        self.n_support = n_support
        self._generate_tasks()

    def _generate_tasks(self):
        offset = 0 if self.train else (self.n_moduli if self.skip == 1 else 1)
        mul = 1 if self.skip == 1 else 2
        moduli = list(range(1 + offset, mul * self.n_moduli + 1 + offset, self.skip))
        if self.n_tasks > self.n_moduli:
            ms = torch.tensor(
                [moduli[i] for i in torch.randint(0, len(moduli), (self.n_tasks,))]
            )
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
        n_support = (
            np.random.randint(10, high=101)
            if self.n_support is None
            else self.n_support
        )
        X_s = torch.randint(0, self.range_max, (n_support, 1)).float()
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
            X_image_s = X_image_s.view(n_support, 32 * 32)
            X_image_q = X_image_q.view(self.range_max, 32 * 32)
        elif self.model in ["lstm", "transformer"]:
            X_image_s = self._image_to_patches(X_image_s)
            X_image_q = self._image_to_patches(X_image_q)
        return (X_image_s, X_s, y_s, X_image_q, X_q, y_q, m)

    def _generate_bits_task(self, m):
        n_support = (
            np.random.randint(10, high=101)
            if self.n_support is None
            else self.n_support
        )
        X_s = torch.randint(0, self.range_max, (n_support, 1)).float()
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
        n_support = (
            np.random.randint(10, high=101)
            if self.n_support is None
            else self.n_support
        )
        X_s = torch.randint(0, self.range_max, (n_support, 1)).float()
        y_s = (X_s % m).float() + torch.normal(0, self.sigma, size=X_s.size())
        X_q = torch.arange(0, self.range_max).float().unsqueeze(1)
        y_q = (X_q % m).float()
        return (X_s, X_s, y_s, X_q, X_s, y_q, m)

    def _generate_bitstrings(self, numbers):
        b = torch.arange(self.bit_width - 1, -1, -1, dtype=torch.int32)
        bitstrings = (numbers >> b) & 1
        return (bitstrings * 2 - 1).unsqueeze(-1).float()


class Omniglot(BaseMetaDataset):

    DATA_PATH = "./omniglot"

    def __init__(
        self,
        n_tasks: int = 10000,
        alphabet: list = None,
        model: str = "cnn",
        N: int = 20,
        K: int = 5,
        train = True,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.model = model
        self.N = N
        self.K = K
        self.train = train
        self.alphabet = alphabet
        self.transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.922059], std=[0.268076])
        ])
        self.transform_test = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.922059], std=[0.268076])
        ])
        self.transform = self.transform_train if self.train else self.transform_test
        self.dataset, self.characters = self._init_dataset()
        self._generate_tasks()

    def _init_dataset(self):
        raw_dataset = datasets.Omniglot(
            root=self.DATA_PATH,
            background=self.train,
            download=True
        )
        images_per_char = []
        for character in raw_dataset._characters:
            char_path = os.path.join(raw_dataset.target_folder, character)
            # list_files returns all *.png in that character directory
            images_per_char.append([
                (file_name,) + os.path.split(character)  
                for file_name in list_files(char_path, ".png")
            ])
        dataset = defaultdict(lambda: defaultdict(list))
        for group in images_per_char:
            for file_name, alpha, character in group:
                dataset[alpha][character].append(file_name)
        # Filter by specified alphabets (if provided and if training)
        if self.alphabet is not None and self.train:
            dataset = {a: dataset[a] for a in self.alphabet if a in dataset}
        characters = [(a, c) for a in dataset for c in dataset[a]]
        return dataset, characters

    def _generate_tasks(self):
        for _ in range(self.n_tasks):
            characters = np.random.choice(self.characters, size=self.N)
            X_s, X_q, y_s, y_q = [], [], [], []
            for i, (alphabet, character) in enumerate(characters):
                images = self.dataset[alphabet][character]
                np.random.shuffle(images)
                support = images[:self.K]
                query = images[self.K:]
                X_s.extend(support)
                y_s.extend([i] * self.K)  # Class label `i` for support set
                X_q.extend(query)
                y_q.extend([i] * len(query))  # Class label `i` for query set
            self.tasks.append((X_s, y_s, X_q, y_q))

    def __getitem__(self, idx):
        X_s, y_s, X_q, y_q = self.tasks[idx]
        X_image_s = []
        for img_path in X_s:
            img_full_path = os.path.join(self.dataset.target_folder, img_path)
            img = Image.open(img_full_path).convert("L")  # grayscale
            img = self.transform(img)
            X_image_s.append(img)
        X_s = torch.stack(X_image_s) 

        # query set
        X_image_q = []
        for img_path in X_q:
            img_full_path = os.path.join(self.dataset.target_folder, img_path)
            img = Image.open(img_full_path).convert("L")
            img = self.transform(img)
            X_image_q.append(img)
        X_q = torch.stack(X_image_q)

        return X_s, y_s, X_q, y_q