import time
import torch
from torch.nn.utils import clip_grad_norm_ as clip
import torch.nn as nn
import numpy as np
import learn2learn as l2l

from evaluation import evaluate, baseline_evaluate
from initialization import init_model
from constants import *


def vanilla_train(
    model,
    train_loader,
    val_dataset,
    test_dataset,
    criterion,
    optimizer,
    device,
    epochs: int = 1000,  # Increased for convergence
    patience: int = 100,  # Increased for convergenc
    adaptation_steps: int = 1,
    verbose: bool = False,
):
    """Train a model using vanilla supervised learning.
    
    Unlike meta_train, this function:
    1. Takes a regular model instead of a meta-learning model
    2. Uses all data (support + query) concatenated into one batch
    3. Performs standard supervised learning updates
    
    Note: Uses the same optimizer (AdamW) and parameters as meta-learning
    for fair comparison.
    """
    if verbose:
        print("--- Vanilla Training ---")
    
    model.train()
    val_losses, test_losses = [], []
    best_val_loss = float("inf")
    best_model_state = None
    no_improve_epochs = 0
    batches_seen = 0
    stop_early = False
    start = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        #TODO: (DONE) concatenate meta-episodes into single batches for pre-training
        for _, (X_s, y_s, X_q, y_q) in enumerate(train_loader):
            # Concatenate all support set examples
            all_X_s = torch.cat([x for x in X_s], dim=0)  
            all_y_s = torch.cat([y for y in y_s], dim=0) 
            #query
            X_q_flat = X_q.reshape(-1, X_q.shape[-1])  #[total_samples, features]
            y_q_flat = y_q.reshape(-1)  # [total_samples]
            
            if len(all_y_s.shape) > 1:
                all_y_s = all_y_s.reshape(-1)
            
            # Combine support and query data
            inputs = torch.cat([all_X_s, X_q_flat], dim=0)
            targets = torch.cat([all_y_s, y_q_flat], dim=0)
            
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(-1, 1)  
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            clip(model.parameters(), 1.0)  
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            batches_seen += 1
            
            if batches_seen % 1000 == 0:  
                val_loss, val_acc = baseline_evaluate(
                    model, val_dataset, criterion, device, [0, adaptation_steps]
                )
                test_loss, test_acc = baseline_evaluate(
                    model, test_dataset, criterion, device, [0, adaptation_steps]
                )
                val_losses.append(val_loss)
                test_losses.append(test_loss)
                no_improve_epochs += 1
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    no_improve_epochs = 0

                if verbose:
                    print(
                        f"Batches {batches_seen}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} ({time.time() - start:.2f}s)"
                    )
                
                if no_improve_epochs >= patience:
                    if verbose:
                        print(
                            f"No validation improvement after {patience} epochs. Stopping early..."
                        )
                    stop_early = True
                    break
                if np.isnan(val_loss):
                    if verbose:
                        print(f"Pre-training diverged. Stopping early...")
                    stop_early = True
                    break
                start = time.time()
        if stop_early:
            break
    return val_losses, test_losses, best_model_state


def meta_train(
    meta,
    train_loader,
    val_dataset,
    test_dataset,
    criterion,
    optimizer,
    device,
    epochs: int = 1,
    tasks_per_meta_batch: int = 4,
    adaptation_steps: int = 1,
    patience=20,  # Number of epochs to wait for improvement
    verbose: bool = False,
):
    if verbose:
        print("--- Meta-Training ---")
    meta.train()
    val_losses, test_losses = [], []
    best_val_loss = float("inf")
    best_model_state = None
    no_improve_epochs = 0
    episodes_seen = 0
    stop_early = False
    start = time.time()
    for _ in range(epochs):
        for i, (X_s, y_s, X_q, y_q) in enumerate(train_loader):
            optimizer.zero_grad()
            meta_loss = 0.0
            for t in range(tasks_per_meta_batch):
                # Adaptation on the support set
                learner = meta.clone()
                for _ in range(adaptation_steps):
                    support_pred = learner(X_s[t])
                    support_loss = criterion(support_pred, y_s[t])
                    learner.adapt(support_loss)
                # Evaluate on the query set
                query_pred = learner(X_q[t])
                query_loss = criterion(query_pred, y_q[t])
                meta_loss += query_loss

            meta_loss /= tasks_per_meta_batch
            meta_loss.backward()
            clip(meta.parameters(), 1.0)
            optimizer.step()
            episodes_seen += len(X_s)
            if episodes_seen % 1000 == 0:
                val_loss, val_acc = evaluate(
                    meta, val_dataset, criterion, device, [0, adaptation_steps]
                )
                test_loss, test_acc = evaluate(
                    meta, test_dataset, criterion, device, [0, adaptation_steps]
                )
                val_losses.append(val_loss)
                test_losses.append(test_loss)
                no_improve_epochs += 1
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = meta.state_dict()
                    no_improve_epochs = 0

                if verbose:
                    print(
                        f"Episodes {episodes_seen}, Meta-Train Loss: {val_loss:.4f}, Meta-Test Loss: {test_loss:.4f}; Meta-Train Acc: {val_acc:.4f}, Meta-Test Acc: {test_acc:.4f} ({time.time() - start:.2f}s) {'*' if best_val_loss == val_loss else ''}"
                    )
                if no_improve_epochs >= patience:
                    if verbose:
                        print(
                            f"No validation improvement after {patience} epochs. Stopping early..."
                        )
                    stop_early = True
                    break
                if np.isnan(val_loss):
                    if verbose:
                        print(f"Meta-training diverged. Stopping early...")
                    stop_early = True
                    break
                start = time.time()
        if stop_early:
            break
    return val_losses, test_losses, best_model_state


def hyper_search(
    experiment,
    m,
    data_type,
    outer_lr,
    train_loader,
    val_dataset,
    test_dataset,
    device,
    epochs: int = 1,
    tasks_per_meta_batch: int = 4,
    adaptation_steps: int = 1,
    channels: int = 1,
    bits: int = 8,
    n_output: int = 1,
):
    print("--- Hyperparameter Search ---")
    best_index, best_val = 0, np.inf
    for index in INDICES:
        start = time.time()
        model = init_model(m, data_type, index, channels=channels, bits=bits, n_output=n_output)
        model = model.to(device)
        meta = l2l.algorithms.MetaSGD(model, lr=1e-3, first_order=False).to(device)
        criterion = nn.MSELoss() if experiment == "mod" else nn.BCEWithLogitsLoss() if experiment == "concept" else nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(meta.parameters(), lr=outer_lr)
        val_losses, _, _ = meta_train(
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
        )
        print(
            f"Results: Model={m}, Parameter Index={index}, Val Loss={min(val_losses):.3f} ({time.time() - start:.2f}s)"
        )
        if min(val_losses) < best_val:
            best_index = index
            best_val = min(val_losses)
    return best_index
