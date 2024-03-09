import torch
import argparse
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from scipy.special import factorial
from torch.nn import ModuleList, Dropout, ReLU
from typing import List

# def get_citation_args():
#     if not hasattr(get_citation_args, "args"):
#         parser = argparse.ArgumentParser()
#         parser.add_argument('--t', type=float, default=3)
#         parser.add_argument('--latestop', action='store_true', default=False)
#         parser.add_argument('--config', type=str, required=True)
#         parser.add_argument('--preprocessing', type=str, default='none')
#         parser.add_argument('--fixT', action='store_true', default=False)
#         parser.add_argument('--debugInfo', action='store_true', default=False)
#         parser.add_argument('--step', type=int, default=10)
#         parser.add_argument('--denseT', action='store_true', default=False)
#         parser.add_argument('--shareT', action='store_true', default=False)
#         parser.add_argument('--lateDiffu', action='store_true', default=False)
#         parser.add_argument('--swapTrainValid', action='store_true', default=False)
#         parser.add_argument('--tLr', type=float, default=0.01)
#         parser.add_argument('--num_per_class', type=int, default=20)
        
#         get_citation_args.args = parser.parse_args()

#         if get_citation_args.args.shareT == True:
#             assert(get_citation_args.args.denseT == True)
            
#     return get_citation_args.args

# class TDPlusConv(MessagePassing):
#     def __init__(self, in_channels, init_t):
#         super(TDPlusConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
#         #args = get_citation_args()
#         self.init_t = init_t
#         self.step = 10
#         if not args.denseT:
#             self.t = Parameter(torch.Tensor(self.step, in_channels))
#         else:
#             self.t = Parameter(torch.Tensor(self.step))
#         # self.t.data.fill_(2)
#         self.reset_parameters()
#         # self.t.requires_grad = False


#     def forward(self, x, edge_index, edge_weight=None):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]
#         # print(self.t)
#         # Step 1: Add self-loops to the adjacency matrix.
#         self.t_norm = torch.nn.functional.softmax(self.t, dim=0)
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         # Step 2: Linearly transform node feature matrix.

#         # Step 3: Compute normalization.
#         row, col = edge_index
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         # Step 4-5: Start propagating messages.
#         x_list = [0 for i in range(self.step)]
#         x_list[0] = x

#         for i in range(1, self.step):
#             x_list[i] = self.propagate(edge_index, x=x_list[i - 1], norm=norm)
        
#         y = 0

#         for k in range(self.step):
#             x_list[k] = self.t_norm[k] * x_list[k] ## important!
#             # x_list[k] = torch.pow(self.t, k) / factorial(k) * x_list[k]
#             if k != 0: 
#                 y += x_list[k]
#             else:
#                 y = x_list[k]
#         return y
    
#     def reset_parameters(self):
#         torch.nn.init.constant_(self.t, self.init_t)
#         #self.t.requires_grad = False
    

#     def message(self, x_j, norm):
#         # x_j has shape [E, out_channels]

#         # Step 4: Normalize node features.
#         return norm.view(-1, 1) * x_j

# class GCNPlusConv(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, init_t):
#         super(GCNPlusConv, self).__init__()
#         self.diffusion = TDPlusConv(in_channels, init_t)
#         self.lin = torch.nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index, edge_weight=None):
#         x = self.diffusion(x, edge_index)
#         x = self.lin(x)
#         return x
    
#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         self.diffusion.reset_parameters()


class TDConv(MessagePassing):
    def __init__(self, in_channels, init_t):
        super(TDConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        # args = get_citation_args()
        self.init_t = init_t
        self.step = 2
        # if not args.denseT:
        # self.t = Parameter(torch.Tensor(in_channels))
        # else:
        self.t = Parameter(torch.Tensor(1))
        # self.t.data.fill_(2)
        self.reset_parameters()
        # self.t.requires_grad = False


    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # print(self.t)
        # Step 1: Add self-loops to the adjacency matrix.
        #self.t_norm = torch.nn.functional.relu(self.t)
        
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        #edge_index = edge_index.to_sparse().indices()
        # Step 2: Linearly transform node feature matrix.

        # Step 3: Compute normalization.
        row, col = edge_index
    
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        x_list = [0 for i in range(self.step)]
        x_list[0] = x
        
        for i in range(1, self.step):
            x_list[i] = self.propagate(edge_index, x=x_list[i - 1], norm=norm)
        
        y = 0

        for k in range(self.step):
            x_list[k] = torch.exp(-self.t) * (torch.pow(self.t, k) / factorial(k)) * x_list[k] ## important!
            #x_list[k] = torch.exp(-self.t_norm) * (torch.pow(self.t_norm, k) / factorial(k)) * x_list[k]
            if k != 0: 
                y += x_list[k]
            else:
                y = x_list[k]
        
        return y
    
    def reset_parameters(self):
        torch.nn.init.constant_(self.t, self.init_t)
        #self.t.requires_grad = False
    

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    
class GCNPlusConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, init_t):
        super(GCNPlusConv, self).__init__()
        self.diffusion = TDConv(in_channels, init_t)
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.diffusion(x, edge_index)
        x = self.lin(x)
        
        return x
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.diffusion.reset_parameters()

# class GCNPlusRConv(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, init_t):
#         super(GCNPlusRConv, self).__init__()
#         self.diffusion = TDConv(out_channels, init_t)
#         self.lin = torch.nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index, edge_weight=None):
#         x = self.lin(x)
#         x = self.diffusion(x, edge_index)
#         return x
    
#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         self.diffusion.reset_parameters()        

class ADC(torch.nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(ADC, self).__init__()
        self.t = 5
        # args = get_citation_args()
        hidden : List[int] = [hid_dim]
        num_features = [in_dim] + hidden + [hid_dim]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            # layers.append(SGConv(in_features, out_features, K=2))
            # if args.lateDiffu:
            #     layers.append(GCNPlusRConv(in_features, out_features, init_t=self.t))
            # else:
            layers.append(GCNPlusConv(in_features, out_features, init_t=self.t))

        self.layers = ModuleList(layers)
        
        self.linear = torch.nn.Linear(adj_dim * hid_dim, adj_dim * hid_dim // 2) # Readout
        self.linear2 = torch.nn.Linear(adj_dim * hid_dim // 2, out_dim) # Predictor
        

        #if args.shareT == True:
        #self.layers[1].diffusion = self.layers[0].diffusion
        # self.reg_params = list(layers[0].parameters())
        # self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    
    def reset_linear(self):
        for layer in self.layers:
            layer.lin.reset_parameters()

    def forward(self, X, A):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        batch = X.shape[0]
        x_list = []
        for b in range(batch):
            edge_index = A[b].to_sparse().indices()
            x = X[b]
            for i, layer in enumerate(self.layers):
                x = layer(x, edge_index)
                
                if i == len(self.layers) - 1:
                    break

                x = self.act_fn(x)
                x = self.dropout(x)
            x_list.append(x)
        x = torch.stack(x_list)
        x = x.reshape(x.shape[0], -1)
        
        x = self.linear(x) # Readout
        x2 = F.relu(x)
        x2 = self.linear2(x2) # Predictor
        
        # return torch.nn.functional.log_softmax(x, dim=1)
        return F.log_softmax(x2, dim=1)
