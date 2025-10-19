import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCGRU(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.cell = GCGRUCell(in_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hid_dim)

        for t in range(T):
            h = self.cell(x[:, t, :], edge_index, h)

        y_hat = self.fc(h)
        return y_hat
    

class GCGRUCell(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # reset
        self.conv_x_r = GCNConv(in_dim, out_dim)
        self.conv_h_r = GCNConv(out_dim, out_dim)
        
        # update
        self.conv_x_z = GCNConv(in_dim, out_dim)
        self.conv_h_z = GCNConv(out_dim, out_dim)
        
        # candidate
        self.conv_x_n = GCNConv(in_dim, out_dim)
        self.conv_h_n = GCNConv(out_dim, out_dim)


    def forward(self, x, edge_index, h_prev):
        r = torch.sigmoid(self.conv_x_r(x, edge_index) + self.conv_h_r(h_prev, edge_index))
        z = torch.sigmoid(self.conv_x_z(x, edge_index) + self.conv_h_z(h_prev, edge_index))
        n = torch.tanh(self.conv_h_n(x, edge_index) + r * self.conv_h_n(h_prev, edge_index))

        h = (1 - z) * h_prev + z * n
        return h