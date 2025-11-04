from GRU import GRU
import argparse
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--t', type=int, default=5, help='Window size')
    p.add_argument('--h-dim', type=int, default=16, help='Hidden dim size')
    p.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    p.add_argument('--train-splits', nargs=3, type=float, default=[0.6, 0.2, 0.2], help='Fractions of data splits')
    return p.parse_args()

def prepare_data(t=1):
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    data = pd.read_csv(url, usecols=[1])
    data = data.values.astype('float32')
    
    data_min, data_max = data.min(), data.max()
    data = (data - data_min) / (data_max - data_min)

    X, y = [], []
    for i in range(data.shape[0] - t - 1):
        a = data[i:(i+t), 0]
        X.append(a)
        y.append(data[i+t, 0])
    return torch.tensor(np.array(X)).unsqueeze(-1), torch.tensor(np.array(y)).unsqueeze(-1), data_min, data_max

def split_data(X, y, fracs=[0.6, 0.2, 0.2]):
    if sum(fracs) > 1:
        raise ValueError(f"Split fractions add up to more than 1: {" + ".join(fracs)} = {sum(fracs)}")
    if len(fracs) != 3:
        raise ValueError("Invalid fraction count")
    train_size = int(len(X) * fracs[0])
    X_train = X[:train_size]
    y_train = y[:train_size]

    val_size = int(len(X) * fracs[1])
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    test_size = int(len(X) * fracs[2])
    X_test = X[train_size+val_size:train_size+val_size+test_size]
    y_test = y[train_size+val_size:train_size+val_size+test_size]
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.01, patience = None):
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()
    epochs = args.epochs

    best_loss = torch.inf
    best_params = None
    patience_counter = 0
    for epoch in range (1, epochs+1):
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

    if patience is not None:
        model.load_state_dict(best_params)
    return model

def plot(pred, true, denorm=False, data_min=None, data_max=None):
    if denorm and (data_min is None or data_max is None):
        raise ValueError(f"Denormalize is true without critical data values provided: data_min={data_min}, data_max={data_max}")
    
    if denorm:
        pred = pred * (data_max - data_min) + data_min
        true = true * (data_max - data_min) + data_min
    
    test_loss = torch.nn.MSELoss()(true, pred)
    plt.plot(true, label='Original Data')
    plt.plot(pred, label='Predicted Data', color='r')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f"Time-Series Forecasting using GRU (test loss = {test_loss.item():.4f})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = get_args()
    X, y, data_min, data_max = prepare_data(t=args.t)
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, args.train_splits)

    model = GRU(1, args.h_dim, 1)
    model = train_model(model, X_train, y_train, X_val, y_val, epochs=args.epochs, lr=0.001, patience=50)
    
    model.eval()
    pred = model(X_test).detach()

    plot(pred, y_test, denorm=True, data_min=data_min, data_max=data_max)
