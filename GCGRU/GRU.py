import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.bias = bias

        self.cell = GRUCell(in_dim, hid_dim, bias)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x: (B, T, in_dim)
        # out: (B, out_dim)
        B, T, _ = x.size()
        h = torch.zeros(B, self.hid_dim)
        for t in range(T):
            h = self.cell(x[:, t, :], h)
        
        y_hat = self.fc(h)
        return y_hat


class GRUCell(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias

        # reset: r_t = σ( x_t @ W_r + h_prev @ U_r )
        # update: z_t = σ( x_t @ W_z + h_prev @ U_z )
        # candidate: n_t = tanh( x_t @ W_n + ( r_t . h_prev ) @ U_n )
        self.x2h = nn.Linear(in_dim, 3*out_dim, bias=bias)
        self.h2h = nn.Linear(out_dim, 3*out_dim, bias=bias)

    def forward(self, x, h_prev):
        # x - shape (B, in_dim); h - shape (B, out_dim)
        x_t = self.x2h(x)
        h_t = self.h2h(h_prev)
        
        x_r, x_z, x_n = x_t.chunk(3, 1)
        h_r, h_z, h_n = h_t.chunk(3, 1)
        
        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)
        n = torch.tanh(x_n + r * h_n)
    
        h = (1 - z) * h_prev + z * n
        return h
