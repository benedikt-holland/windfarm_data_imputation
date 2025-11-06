import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from .dataset import Item

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def train_epoch(
    model: nn.Module,
    batch: Item,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,    
):
    model.train()
    optimizer.zero_grad()

    preds = model(batch.X)

    # Pure imputation task -> loss based on predicting simulated missing values correctly
    loss_mask = (batch.nan_mask == 1) & (batch.data_mask == 0)
    loss = criterion(preds[loss_mask], batch.y[loss_mask])

    loss.backward()
    optimizer.step()
    return loss.item()

def validate_epoch(
    model: nn.Module,
    batch: Item,
    criterion: nn.Module,            
):
    model.eval()
    with torch.no_grad():
        preds = model(batch.X)

        # Pure imputation task -> loss based on predicting simulated missing values correctly
        loss_mask = (batch.nan_mask == 1) & (batch.data_mask == 0)
        loss = criterion(preds[loss_mask], batch.y[loss_mask])
    
    return loss.item()

def train(
    model: nn.Module,
    train_data: torch.utils.data.DataLoader | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: torch.utils.data.DataLoader | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
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
                train_loss = train_epoch(model, batch, optimizer, criterion)
                epoch_train_loss += train_loss * len(batch.X)
            
            epoch_train_loss /= len(train_data.dataset)
        elif isinstance(train_data, tuple):
            train_X, train_y, train_nan_mask, train_data_mask = train_data
            epoch_train_loss = train_epoch(model, Item(X=train_X, y=train_y, nan_mask=train_nan_mask, data_mask=train_data_mask), optimizer, criterion)

        if isinstance(val_data, torch.utils.data.DataLoader):
            for batch in val_data:
                val_loss = validate_epoch(model, batch, criterion)
                epoch_val_loss += val_loss * len(batch.X)
            epoch_val_loss /= len(val_data.dataset)
        elif isinstance(val_data, tuple):
            val_X, val_y, val_nan_mask, val_data_mask = val_data
            epoch_val_loss = validate_epoch(model, Item(X=val_X, y=val_y, nan_mask=val_nan_mask, data_mask=val_data_mask), criterion)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch: {epoch};\ttraining loss: {epoch_train_loss:.4f};\tvalidation loss: {epoch_val_loss:.4f}")

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

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

    if best_params is not None:
        model.load_state_dict(best_params)

    return model, train_losses, val_losses