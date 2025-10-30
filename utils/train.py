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

    preds = model(X_batch)  # [B, T, 1]

    # y_batch is shape [B, T, 3], where in the last dim, 
    # idx 0 is the true pred, idx 1 is the data mask (0 if unseen), 
    # and idx 2 is the nan mask (0 if nan)
    y_true = y_batch[..., 0].unsqueeze(-1)
    data_mask = y_batch[..., 1]
    nan_mask = y_batch[..., 2]

    # Pure imputation task -> loss based on predicting simulated missing values correctly
    loss_mask = (nan_mask == 1) & (data_mask == 0)
    loss = criterion(preds[loss_mask], y_true[loss_mask])

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
    with torch.no_grad():
        preds = model(X_batch)

        # y_batch is shape [B, T, 3], where in the last dim, 
        # idx 0 is the true pred, idx 1 is the data mask (0 if unseen), 
        # and idx 2 is the nan mask (0 if nan)
        y_true = y_batch[..., 0].unsqueeze(-1)
        data_mask = y_batch[..., 1]
        nan_mask = y_batch[..., 2]

        # Pure imputation task -> loss based on predicting simulated missing values correctly
        loss_mask = (nan_mask == 1) & (data_mask == 0)
        loss = criterion(preds[loss_mask], y_true[loss_mask])
    
    return loss.item()

def train(
    model: nn.Module,
    train_data: torch.utils.data.DataLoader | tuple[torch.Tensor, torch.Tensor],
    val_data: torch.utils.data.DataLoader | tuple[torch.Tensor, torch.Tensor],
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

        if isinstance(train_data, torch.utils.data.DataLoader):
            for batch in train_data:
                train_loss = train_epoch(model, batch.X, batch.y, optimizer, criterion)
                epoch_train_loss += train_loss * len(batch.X)
            
            epoch_train_loss /= len(train_data.dataset)
        elif isinstance(train_data, tuple):
            train_X, train_y = train_data
            epoch_train_loss = train_epoch(model, train_X, train_y, optimizer, criterion)

        if isinstance(val_data, torch.utils.data.DataLoader):
            for batch in val_data:
                val_loss = validate_epoch(model, batch.X, batch.y, criterion)
                epoch_val_loss += val_loss * len(batch.X)
            epoch_val_loss /= len(val_data.dataset)
        elif isinstance(val_data, tuple):
            val_X, val_y = val_data
            epoch_val_loss = validate_epoch(model, val_X, val_y, criterion)
        
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