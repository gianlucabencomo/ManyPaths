import os
import random
import numpy as np
import torch


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