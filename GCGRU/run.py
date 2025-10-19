from GRU import GRU
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--t', type=int, default=5, help='Window size')
    p.add_argument('--h-dim', type=int, default=16, help='Hidden dim size')
    p.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    p.add_argument('--train-split', type=float, default=0.8, help='Fraction of data to use for training')
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

def split_data(X, y, frac=0.8):
    train_size = int(len(X) * frac)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, y_train, X_test, y_test

def train_model(model, X, y, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()
    epochs = args.epochs

    for epoch in range (1, epochs+1):
        model.train()
        optimizer.zero_grad()
        
        pred = model(X)
        train_loss = criterion(y, pred)
        
        train_loss.backward()
        optimizer.step()
        
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch: {epoch};\t training loss: {train_loss.item():.4f}")

    return model

def plot(pred, true, denorm=False, data_min=None, data_max=None):
    if denorm and (data_min is None or data_max is None):
        raise ValueError(f"Denormalize is true without critical data values provided: data_min={data_min}, data_max={data_max}")
    
    if denorm:
        pred = pred * (data_max - data_min) + data_min
        true = true * (data_max - data_min) + data_min
    
    plt.plot(true, label='Original Data')
    plt.plot(pred, label='Predicted Data', color='r')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time-Series Forecasting using GRU')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = get_args()
    X, y, data_min, data_max = prepare_data(t=args.t)
    
    X_train, y_train, X_test, y_test = split_data(X, y, args.train_split)

    model = GRU(1, args.h_dim, 1)
    model = train_model(model, X_train, y_train, epochs=args.epochs)
    
    model.eval()
    pred = model(X_test).detach()

    plot(pred, y_test, denorm=True, data_min=data_min, data_max=data_max)
