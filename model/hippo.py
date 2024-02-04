import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import nengo
from scipy import signal
from scipy import linalg as la
from scipy import special as ss

class HiPPO_LegT(nn.Module):
    def __init__(
        self,
        N,
        dt=1.0,
        method='bilinear'
    ):
        super().__init__()
        self.N = N
        A, B = transition('lmu', N)
        C = np.ones((1, N))
        D = np.zeros((1, ))
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=method)
        
        B = B.squeeze(-1)
        
        self.register_buffer('A', torch.Tensor(A))
        self.register_buffer('B', torch.Tensor(B))
        
        vals = np.arange(0.0, 1.0, dt)
        self.eval_matrix = torch.Tensor(ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T)
    
    def forward(self, inputs):
        inputs = inputs.unsqueeze(-1)
        u = inputs * self.B #   Control?
        
        c = torch.zeros(u.shape[1:])
        cs = []
        for f in inputs:
            c = F.linear(c, self.A) + self.B * F
            cs.append(c)
        return torch.stack(cs, dim=0)
    
    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)
        
        