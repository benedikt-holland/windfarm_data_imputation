import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def train_epoch(
    model: nn.Module,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,    
):
    model.train()
    optimizer.zero_grad()

    preds = model(X_batch)
    loss = criterion(y_batch, preds)

    loss.backward()
    optimizer.step()
    return loss.item()

def validate_epoch(
    model: nn.Module,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    criterion: nn.Module,            
):
    model.eval()

    preds = model(X_batch)
    loss = criterion(y_batch, preds)
    
    return loss.item()

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 100,
    patience: int | None = None
):
    train_losses = []
    val_losses = []

    best_loss = torch.inf
    best_params = None
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        epoch_train_loss = 0
        epoch_val_loss = 0

        for batch in train_loader:
            train_loss = train_epoch(model, batch["X"], batch["y"], optimizer, criterion)
            epoch_train_loss += train_loss * len(batch["X"])
        
        epoch_train_loss /= len(train_loader.dataset)
        
        for batch in val_loader:
            val_loss = validate_epoch(model, batch["X"], batch["y"], criterion)
            epoch_val_loss += val_loss * len(batch["X"])
        epoch_val_loss /= len(val_loader.dataset)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch: {epoch};\ttraining loss: {epoch_train_loss:.4f};\tvalidation loss: {epoch_val_loss:.4f}")

        if patience is not None:
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_params = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Stopping early on epoch {epoch}")
                    break
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

    if best_params is not None:
        model.load_state_dict(best_params)

    return model, train_losses, val_losses