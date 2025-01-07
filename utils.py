import os
import random
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

def collate_concept(batch, device='cpu'):
    X_s, y_s, X_q, y_q = zip(*batch)

    # Move consistent sizes to the device
    X_q = torch.stack(X_q).to(device)
    y_q = torch.stack(y_q).to(device)

    # For support sets (if sizes vary), move each to device individually
    X_s = [x.to(device) for x in X_s]
    y_s = [y.to(device) for y in y_s]

    return X_s, y_s, X_q, y_q

def collate_default(batch, device='cpu'):
    def move_to_device(data):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, list):
            return [move_to_device(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(move_to_device(item) for item in data)
        elif isinstance(data, dict):
            return {key: move_to_device(value) for key, value in data.items()}
        else:
            return data
    batch = default_collate(batch)
    return move_to_device(batch)

def get_collate(experiment: str, device='cpu'):
    if experiment == "concept":
        return lambda batch: collate_concept(batch, device=device)
    else:
        return lambda batch: collate_default(batch, device=device)

def set_random_seeds(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 1 gpu

def save_model(meta, save_dir="state_dicts", file_prefix="meta_learning"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(meta.state_dict(), f"{save_dir}/{file_prefix}.pth")
    print(f"Model saved to {save_dir}/{file_prefix}.pth")

def calculate_accuracy(predictions, targets):
    predictions = (predictions > 0.0).float()
    # Compare with targets and compute accuracy
    correct = (predictions == targets).sum().item()
    accuracy = correct / targets.numel()
    return accuracy