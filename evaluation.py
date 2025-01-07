import torch
import numpy as np
from utils import calculate_accuracy


def evaluate(meta, dataset, criterion, device, adaptation_steps, return_results=False):
    meta.train()
    meta_loss, meta_acc, results = 0.0, [], []
    for task in dataset.tasks:
        X_s, X_num_s, y_s, X_q, X_num_q, y_q, m = task
        X_s, y_s, X_q, y_q = (
            X_s.to(device),
            y_s.to(device),
            X_q.to(device),
            y_q.to(device),
        )
        preds, losses, accs = [], [], []
        for steps in adaptation_steps:
            learner = meta.clone()
            # Adaptation on the support set
            for _ in range(steps):
                support_pred = learner(X_s)
                support_loss = criterion(support_pred, y_s)
                learner.adapt(support_loss)

            # Evaluate on the query set (post-adaptation)
            with torch.no_grad():
                pred = learner(X_q)
                preds.append(pred)
                losses.append(criterion(pred, y_q).item())
                accs.append(calculate_accuracy(pred, y_q))

        results.append(
            {
                "m": m,
                "X_s": X_num_s,
                "y_s": y_s,
                "X_q": X_num_q,
                "y_q": y_q,
                "predictions": preds,
                "losses": losses,
                "accuracies": accs,
            }
        )
        meta_acc.append(accs[:][1])
        meta_loss += losses[1]

    meta_loss /= len(dataset.tasks)
    meta_acc = np.mean(meta_acc)
    if return_results:
        return meta_loss, results
    else:
        return meta_loss, meta_acc
