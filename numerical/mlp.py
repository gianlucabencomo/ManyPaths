import typer

import torch
import torch.nn as nn
import learn2learn as l2l
import matplotlib.pyplot as plt

from datasets import MetaNumberModuloDataset
from models import MLP
from utils import set_random_seeds
from visualize import plot_loss, plot_meta_test_results


def meta_train(
    meta,
    dataset,
    criterion,
    optimizer,
    device,
    epochs: int = 10000,
    tasks_per_meta_batch: int = 4,
    adaptation_steps: int = 1,
):
    print("--- Meta-Training ---")
    meta.train()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        meta_loss = 0.0

        for _ in range(tasks_per_meta_batch):
            # Sample a task
            X_s, y_s, X_q, y_q, _ = dataset.sample_task()
            X_s, y_s, X_q, y_q = (
                X_s.to(device),
                y_s.to(device),
                X_q.to(device),
                y_q.to(device),
            )

            # Adaptation on the support set
            learner = meta.clone()
            for _ in range(adaptation_steps):
                support_pred = learner(X_s)
                support_loss = criterion(support_pred, y_s)
                learner.adapt(support_loss)

            # Evaluate on the query set
            query_pred = learner(X_q)
            query_loss = criterion(query_pred, y_q)
            meta_loss += query_loss

        # Meta-update
        meta_loss /= tasks_per_meta_batch
        meta_loss.backward()
        optimizer.step()

        losses.append(meta_loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Meta Loss: {meta_loss.item():.4f}")

    return losses

def meta_test(meta, dataset, criterion, device, n_tasks, adaptation_steps):
    print("\n--- Meta-Testing ---")
    meta.train()
    results = []
    for t in range(n_tasks):
        X_s, y_s, X_q, y_q, m = dataset.tasks[t]
        X_s, y_s, X_q, y_q = X_s.to(device), y_s.to(device), X_q.to(device), y_q.to(device)

        learner = meta.clone()
        # Evaluate on the query set (pre-adaptation)
        with torch.no_grad():
            pred_no_adap = learner(X_q)
            pre_loss = criterion(pred_no_adap, y_q)

        # Adaptation on the support set
        for _ in range(adaptation_steps):
            support_pred = learner(X_s)
            support_loss = criterion(support_pred, y_s)
            learner.adapt(support_loss)

        # Evaluate on the query set (post-adaptation)
        with torch.no_grad():
            pred_w_adap = learner(X_q)
            post_loss = criterion(pred_w_adap, y_q)
        
        results.append({
            "m": m,
            "X_q": X_q,
            "y_q": y_q,
            "pre_adaptation_loss": pre_loss.item(),
            "post_adaptation_loss": post_loss.item(),
            "pred_no_adap": pred_no_adap,
            "pred_w_adap": pred_w_adap,
        })
        print(f"Task {t+1}/{n_tasks} - Pre-adaptation Loss: {pre_loss.item():.4f}, Post-adaptation Loss: {post_loss.item():.4f}")

    return results

def main(
    seed: int = 0,
    n_tasks: int = 20,
    n_samples_per_task: int = 20,
    epochs: int = 1000,
    tasks_per_meta_batch: int = 4,
    adaptation_steps: int = 1,
    inner_lr: float = 1e-3,
    outer_lr: float = 1e-3,
    use_meta_sgd: bool = False
):
    set_random_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init datset
    dataset = MetaNumberModuloDataset(n_tasks, n_samples_per_task, range_max=100)

    # init model, meta-learner, loss, and meta-optimizer
    model = MLP().to(device)
    if use_meta_sgd:
        meta = l2l.algorithms.MetaSGD(model, lr=inner_lr, first_order=False).to(device)
    else:
        meta = l2l.algorithms.MAML(model, lr=inner_lr, first_order=False).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(meta.parameters(), lr=outer_lr)

    # meta-training
    losses = meta_train(meta, dataset, criterion, optimizer, device, epochs, tasks_per_meta_batch, adaptation_steps)

    plot_loss(losses)

    # meta-testing
    results = meta_test(meta, dataset, criterion, device, n_tasks, adaptation_steps)

    plot_meta_test_results(results)

    plt.show()

if __name__ == '__main__':
    typer.run(main)
