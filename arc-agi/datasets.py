
class ARCDataset(Dataset):
    def __init__(self, n_tasks, n_samples_per_task):
        self.n_tasks = n_tasks
        self.n_samples_per_task = n_samples_per_task
        self.tasks = []
        self._generate_tasks()

    def __len__(self):
        return len(self.tasks)
        