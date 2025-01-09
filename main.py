import typer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import learn2learn as l2l
import matplotlib.pyplot as plt

from training import meta_train, hyper_search
from evaluation import evaluate
from initialization import init_dataset, init_model
from utils import set_random_seeds, save_model, get_collate
from visualize import plot_loss, plot_meta_test_results
from constants import *


def main(
    seed: int = 0,
    experiment: str = "mod",  # ["mod", "concept"]
    m: str = "mlp",  # ['mlp', 'cnn', 'lstm', 'transformer']
    data_type: str = "image",  # ['image', 'bits', 'number']
    epochs: int = 1000,  # until convergence
    tasks_per_meta_batch: int = 4,
    adaptation_steps: int = 1,
    outer_lr: float = 1e-3,
    skip: int = 1,
    no_hyper_search: bool = False,
    plot: bool = False,
    save: bool = False,
):
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else ("cpu" if m in ["mlp", "lstm"] else "mps")
    )
    print(f"Device: {device}")

    set_random_seeds(seed)
    # init dataset
    collate_fn = get_collate(experiment, device)
    train_dataset, test_dataset, val_dataset = init_dataset(
        experiment, m, data_type, skip
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=tasks_per_meta_batch,
        shuffle=True,
        drop_last=True,
        pin_memory=(device == "cuda:0"),
        collate_fn=collate_fn,
    )

    channels = 3 if experiment == "concept" else 1
    bits = 4 if experiment == "concept" else 8
    if no_hyper_search:
        index = DEFAULT_INDEX
    else:
        index = hyper_search(
            experiment,
            m,
            data_type,
            outer_lr,
            train_loader,
            val_dataset,
            test_dataset,
            device,
            channels=channels,
            bits=bits,
        )

    # init meta-learner, loss, and meta-optimizer
    model = init_model(
        m, data_type, index=index, verbose=True, channels=channels, bits=bits
    ).to(device)
    meta = l2l.algorithms.MetaSGD(model, lr=1e-3, first_order=False).to(device)
    criterion = nn.MSELoss() if experiment == "mod" else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(meta.parameters(), lr=outer_lr)

    # meta-training
    train_losses, test_losses, state_dict = meta_train(
        meta,
        train_loader,
        val_dataset,
        test_dataset,
        criterion,
        optimizer,
        device,
        epochs,
        tasks_per_meta_batch,
        adaptation_steps,
        verbose=True,
    )
    if plot:
        plot_loss(train_losses, test_losses)

    # load best model + save
    if save:
        if experiment == "mod":
            file_prefix = (
                experiment
                + "_"
                + m
                + "_"
                + str(index)
                + "_"
                + data_type
                + "_"
                + str(skip)
                + "_"
                + str(seed)
            )
        else:
            file_prefix = (
                experiment
                + "_"
                + m
                + "_"
                + str(index)
                + "_"
                + data_type
                + "_"
                + str(seed)
            )
        meta.load_state_dict(state_dict)
        save_model(meta, file_prefix=file_prefix)

    if experiment == "mod" and plot:
        _, results = evaluate(
            meta,
            val_dataset,
            criterion,
            device,
            [0, adaptation_steps],
            return_results=True,
        )
        plot_meta_test_results(results)
        _, results = evaluate(
            meta,
            test_dataset,
            criterion,
            device,
            [0, adaptation_steps],
            return_results=True,
        )
        plot_meta_test_results(results)
        plt.show()


if __name__ == "__main__":
    typer.run(main)
