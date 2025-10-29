import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def train(
        model: nn.Module,
        X_train: torch.Tensor | pd.DataFrame | np.ndarray,
        y_train: torch.Tensor | pd.DataFrame | np.ndarray,
        X_val: torch.Tensor | pd.DataFrame | np.ndarray,
        y_val: torch.Tensor | pd.DataFrame | np.ndarray,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int = 100,
        patience: int | None = None):
    train_losses = []
    val_losses = []

    best_loss = torch.inf
    best_params = None
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        train_pred = model(X_train)
        train_loss = criterion(y_train, train_pred)
        
        train_loss.backward()
        optimizer.step()

        model.eval()
        val_pred = model(X_val)
        val_loss = criterion(y_val, val_pred)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch: {epoch};\ttraining loss: {train_loss.item():.4f};\tvalidation loss: {val_loss.item():.4f}")

        if patience is not None:
            if val_loss < best_loss:
                best_loss = val_loss.item()
                best_params = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Stopping early on epoch {epoch}")
                    break
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    if best_params is not None:
        model.load_state_dict(best_params)

    return model, train_losses, val_losses