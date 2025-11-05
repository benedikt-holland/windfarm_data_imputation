import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCGRU(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, edge_index, edge_weight = None):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        self.cell = GCGRUCell(in_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        B, T, N, _ = x.shape  # batch, time, nodes, (features)
        h = torch.zeros(B, N, self.hid_dim, device=x.device)

        outs = []
        for t in range(T):
            h = self.cell(x[:, t, :, :], self.edge_index, h, edge_weight=self.edge_weight)
            outs.append(self.fc(h))  # (B, N, out_dim)

        return torch.stack(outs, dim=1)  # (B, T, N, out_dim)
    

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


    def forward(self, x, edge_index, h_prev, edge_weight = None):
        B, N, _ = x.shape  # batches, nodes, (features)
        x = x.reshape(B * N, -1)
        h_prev = h_prev.reshape(B * N, -1)

        edge_index = edge_index.repeat(1, B) + torch.arange(B, device=x.device).repeat_interleave(edge_index.shape[1]) * N
        if edge_weight is not None:
            edge_weight = edge_weight.repeat(B)

        r = torch.sigmoid(self.conv_x_r(x, edge_index, edge_weight=edge_weight) + self.conv_h_r(h_prev, edge_index, edge_weight=edge_weight))
        z = torch.sigmoid(self.conv_x_z(x, edge_index, edge_weight=edge_weight) + self.conv_h_z(h_prev, edge_index, edge_weight=edge_weight))
        n = torch.tanh(self.conv_x_n(x, edge_index, edge_weight=edge_weight) + r * self.conv_h_n(h_prev, edge_index, edge_weight=edge_weight))

        h = (1 - z) * h_prev + z * n
        h = h.view(B, N, self.out_dim)
        return h