import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def gdc(A, alpha=0.05, eps=1e-5):
    N = A.shape[0]

    ### Self-loops
    A_loop = torch.eye(N) + A

    ### Symmetric transition matrix
    D_loop_vec = A_loop.sum(0).flatten(-1)
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = torch.diag(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

    ### PPR-based diffusion
    S = alpha * torch.linalg.inv(torch.eye(N) - (1 - alpha) * T_sym)

    ### Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)

    ### Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(0).flatten(-1)
    T_S = S_tilde / D_tilde_vec

    return T_S


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        return output


class GraphConvolution(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GraphConvolution, self).__init__()
        self.gc1 = GraphConvolutionLayer(in_dim, hid_dim)
        self.gc2 = GraphConvolutionLayer(hid_dim, out_dim)
        self.dropout = dropout
        self.f = None
        self.rdp = None
        self.rdp2 = None
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, adj):
        tmp = self.gc1(x, adj)
        x2 = F.relu(tmp)

        with torch.no_grad():
            x_one = torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.where(tmp <= 0., x_zero, x_one)
            tmp = x2

        x2 = F.dropout(x2, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.mul(self.rdp, torch.where((tmp != 0.) & (x2 == 0.), x_zero, x_one))

        self.f = x2
        x3 = self.gc2(x2, adj)

        if self.training:
            self.final_conv_acts = x3
            self.final_conv_acts.register_hook(self.activations_hook)

        x4 = F.relu(x3)

        with torch.no_grad():
            x_one = torch.ones_like(x3)
            x_zero = torch.zeros_like(x3)
            self.rdp2 = torch.where(x4 <= 0., x_zero, x_one)

        return x4


class GDC(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(GDC, self).__init__()
        self.dropout = dropout
        self.gcn = GraphConvolution(in_dim, hid_dim, hid_dim, dropout)
        self.linear = nn.Linear(adj_dim * hid_dim, adj_dim * hid_dim // 2)
        self.linear2 = nn.Linear(adj_dim * hid_dim // 2, out_dim)
        self.linrdp = None

    def forward(self, x, hk):
        xs = x.shape[0]
        x = self.gcn(x, hk).reshape(xs, -1)

        x = self.linear(x)
        x2 = F.relu(x)

        with torch.no_grad():
            x_one = torch.ones_like(x)
            x_zero = torch.zeros_like(x)
            self.linrdp = torch.where(x2 <= 0., x_zero, x_one)

        x2 = self.linear2(x2)

        return F.log_softmax(x2, dim=1)