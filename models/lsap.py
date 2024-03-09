import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

### Layer 1
class GraphConvolution1(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self): # Initialize weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj): # Graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)

        return output

class GCN1(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution1(in_dim, out_dim) 
        self.dropout = dropout
        self.f = None
        self.rdp = None
        self.rdp2 = None
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, adj): # Graph convolution part 
        ### Graph convolution 1
        tmp = self.gc1.forward(x, adj)
        x2 = F.relu(tmp)

        with torch.no_grad():
            x_one = torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.where(tmp <= 0., x_zero, x_one)
            tmp = x2
            
        self.f = x
        
        

        return x2

class LSAP1(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(LSAP1, self).__init__()
        self.dropout = dropout
        self.gcn = GCN1(in_dim, hid_dim, hid_dim, dropout) # Graph convolution
        self.linear = nn.Linear(adj_dim * hid_dim, adj_dim * hid_dim // 2) # Readout 1
        self.linear2 = nn.Linear(adj_dim * hid_dim // 2, out_dim) # Readout 2
        self.linrdp = None
        self.linrdp2 = None

    """
    x: (# subjects, # ROI features, # used features)
    adj: (# subjects, # ROI features, # ROI features)
    """
    def forward(self, x, adj): 
        xs = x.shape[0] # (# subjects)
        x = self.gcn.forward(x, adj).reshape(xs, -1) # Graph convolution

        x1 = self.linear(x) # Readout 1
        x2 = F.relu(x1)

        with torch.no_grad():
            x_one = torch.ones_like(x1)
            x_zero = torch.zeros_like(x1)
            self.linrdp = torch.where(x2 <= 0., x_zero, x_one)

        x3 = self.linear2(x2) # Readout 2
        x4 = x3
        
        with torch.no_grad():
            x_one = torch.ones_like(x3)
            x_zero = torch.zeros_like(x3)
            self.linrdp2 = torch.where(x4 <= 0., x_zero, x_one)
        
        return F.log_softmax(x4, dim=1)



### Layer 2
class GraphConvolution2(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self): # Initialize weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj): # Graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)

        return output

class GCN2(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN2, self).__init__()
        self.gc1 = GraphConvolution2(in_dim, hid_dim) 
        self.gc2 = GraphConvolution2(hid_dim, out_dim) 
        self.dropout = dropout
        self.f = None
        self.rdp = None
        self.rdp2 = None
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, adj): # Graph convolution part 
        ### Graph convolution 1
        tmp = self.gc1.forward(x, adj)
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
        
        ### Graph convolution 2
        x3 = self.gc2.forward(x2, adj)

        if self.training:
            self.final_conv_acts = x3
            self.final_conv_acts.register_hook(self.activations_hook)

        x4 = F.relu(x3)

        with torch.no_grad():
            x_one = torch.ones_like(x3)
            x_zero = torch.zeros_like(x3)
            self.rdp2 = torch.where(x4 <= 0., x_zero, x_one)

        return x4

class LSAP2(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(LSAP2, self).__init__()
        self.dropout = dropout
        self.gcn = GCN2(in_dim, hid_dim, hid_dim, dropout) # Graph convolution
        self.linear = nn.Linear(adj_dim * hid_dim, adj_dim * hid_dim // 2) # Readout 1
        self.linear2 = nn.Linear(adj_dim * hid_dim // 2, out_dim) # Readout 2
        self.linrdp = None
        self.linrdp2 = None

    """
    x: (# subjects, # ROI features, # used features)
    adj: (# subjects, # ROI features, # ROI features)
    """
    def forward(self, x, adj): 
        xs = x.shape[0] # (# subjects)
        x = self.gcn.forward(x, adj).reshape(xs, -1) # Graph convolution

        x1 = self.linear(x) # Readout 1
        x2 = F.relu(x1)

        with torch.no_grad():
            x_one = torch.ones_like(x1)
            x_zero = torch.zeros_like(x1)
            self.linrdp = torch.where(x2 <= 0., x_zero, x_one)

        x3 = self.linear2(x2) # Readout 2
        x4 = x3
        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            x_one = torch.ones_like(x3)
            x_zero = torch.zeros_like(x3)
            self.linrdp2 = torch.where(x4 <= 0., x_zero, x_one)
        
        return F.log_softmax(x4, dim=1)
    
    
    
### Layer 3
class GraphConvolution3(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self): # Initialize weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj): # Graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)

        return output


class GCN3(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN3, self).__init__()
        self.gc1 = GraphConvolution3(in_dim, hid_dim) 
        self.gc2 = GraphConvolution3(hid_dim, hid_dim) 
        self.gc3 = GraphConvolution3(hid_dim, out_dim) 
        self.dropout = dropout
        self.f = None
        self.f2 = None
        self.rdp = None
        self.rdp2 = None
        self.rdp3 = None
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, adj): # Graph convolution part 
        ### Graph convolution 1
        tmp = self.gc1.forward(x, adj)
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
        
        ### Graph convolution 2
        x3 = self.gc2.forward(x2, adj)

        if self.training:
            self.final_conv_acts = x3
            self.final_conv_acts.register_hook(self.activations_hook)

        x4 = F.relu(x3)

        with torch.no_grad():
            x_one = torch.ones_like(x3)
            x_zero = torch.zeros_like(x3)
            self.rdp2 = torch.where(x4 <= 0., x_zero, x_one)
            x3 = x4
            
        x4 = F.dropout(x4, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x4)
            x_zero = torch.zeros_like(x4)
            self.rdp = torch.mul(self.rdp2, torch.where((x3 != 0.) & (x4 == 0.), x_zero, x_one))

        self.f2 = x4
        
        ### Graph convolution 3
        x5 = self.gc3.forward(x4, adj)
        #x6 = F.relu(x5)
        x6 = x5

        if self.training:
            self.final_conv_acts = x5
            self.final_conv_acts.register_hook(self.activations_hook)

        with torch.no_grad():
            x_one = torch.ones_like(x6)
            x_zero = torch.zeros_like(x6)
            self.rdp3 = torch.where(x6 <= 0., x_zero, x_one)

        return x6


class LSAP3(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(LSAP3, self).__init__()
        self.dropout = dropout
        self.gcn = GCN3(in_dim, hid_dim, hid_dim, dropout) # Graph convolution
        self.linear = nn.Linear(adj_dim * hid_dim, adj_dim * hid_dim // 2) # Readout 1
        self.linear2 = nn.Linear(adj_dim * hid_dim // 2, out_dim) # Readout 2
        self.linrdp = None
        self.linrdp2 = None

    """
    x: (# subjects, # ROI features, # used features)
    adj: (# subjects, # ROI features, # ROI features)
    """
    def forward(self, x, adj): 
        xs = x.shape[0] # (# subjects)
        x = self.gcn.forward(x, adj).reshape(xs, -1) # Graph convolution

        x1 = self.linear(x) # Readout 1
        x2 = F.relu(x1)

        with torch.no_grad():
            x_one = torch.ones_like(x1)
            x_zero = torch.zeros_like(x1)
            self.linrdp = torch.where(x2 <= 0., x_zero, x_one)

        x3 = self.linear2(x2) # Readout 2
        x4 = x3
        
        with torch.no_grad():
            x_one = torch.ones_like(x3)
            x_zero = torch.zeros_like(x3)
            self.linrdp2 = torch.where(x4 <= 0., x_zero, x_one)
        
        return F.log_softmax(x4, dim=1)
    
    

### 4 Layer
class GraphConvolution4(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution4, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self): # Initialize weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj): # Graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)

        return output

class GCN4(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN4, self).__init__()
        self.gc1 = GraphConvolution4(in_dim, hid_dim) 
        self.gc2 = GraphConvolution4(hid_dim, hid_dim) 
        self.gc3 = GraphConvolution4(hid_dim, hid_dim)
        self.gc4 = GraphConvolution4(hid_dim, out_dim)
        self.dropout = dropout
        self.f = None
        self.f2 = None
        self.f3 = None
        self.rdp = None
        self.rdp2 = None
        self.rdp3 = None
        self.rdp4 = None
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, adj): # Graph convolution part 
        ### Graph convolution 1
        x1 = self.gc1.forward(x, adj)
        x2 = F.relu(x1)

        with torch.no_grad():
            x_one = torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.where(x1 <= 0., x_zero, x_one)
            x1 = x2

        x2 = F.dropout(x2, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.mul(self.rdp, torch.where((x1 != 0.) & (x2 == 0.), x_zero, x_one))

        self.f = x2
        
        ### Graph convolution 2
        x3 = self.gc2.forward(x2, adj)
        x4 = F.relu(x3)

        if self.training:
            self.final_conv_acts = x3
            self.final_conv_acts.register_hook(self.activations_hook)
            
        with torch.no_grad():
            x_one = torch.ones_like(x4)
            x_zero = torch.zeros_like(x4)
            self.rdp2 = torch.where(x4 <= 0., x_zero, x_one)
            x3 = x4

        x4 = F.dropout(x4, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x4)
            x_zero = torch.zeros_like(x4)
            self.rdp = torch.mul(self.rdp2, torch.where((x3 != 0.) & (x4 == 0.), x_zero, x_one))

        self.f2 = x4
        
        ### Graph convolution 3
        x5 = self.gc3.forward(x4, adj)
        x6 = F.relu(x5)
        
        if self.training:
            self.final_conv_acts = x5
            self.final_conv_acts.register_hook(self.activations_hook)

        with torch.no_grad():
            x_one = torch.ones_like(x6)
            x_zero = torch.zeros_like(x6)
            self.rdp3 = torch.where(x6 <= 0., x_zero, x_one)
            x5 = x6
            
        x6 = F.dropout(x6, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x6)
            x_zero = torch.zeros_like(x6)
            self.rdp2 = torch.mul(self.rdp3, torch.where((x5 != 0.) & (x6 == 0.), x_zero, x_one))

        self.f3 = x6
        
        ### Graph convolution 4
        x7 = self.gc4.forward(x6, adj)
        x8 = x7
        
        if self.training:
            self.final_conv_acts = x7
            self.final_conv_acts.register_hook(self.activations_hook)

        with torch.no_grad():
            x_one = torch.ones_like(x8)
            x_zero = torch.zeros_like(x8)
            self.rdp4 = torch.where(x8 <= 0., x_zero, x_one)

        return x8

class LSAP4(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(LSAP4, self).__init__()
        self.dropout = dropout
        self.gcn = GCN4(in_dim, hid_dim, hid_dim, dropout) # Graph convolution
        self.linear = nn.Linear(adj_dim * hid_dim, adj_dim * hid_dim // 2) # Readout 1
        self.linear2 = nn.Linear(adj_dim * hid_dim // 2, out_dim) # Readout 2
        self.linrdp = None
        self.linrdp2 = None

    """
    x: (# subjects, # ROI features, # used features)
    adj: (# subjects, # ROI features, # ROI features)
    """
    def forward(self, x, adj): 
        xs = x.shape[0] # (# subjects)
        x = self.gcn.forward(x, adj).reshape(xs, -1) # Graph convolution

        x1 = self.linear(x) # Readout 1
        x2 = F.relu(x1)

        with torch.no_grad():
            x_one = torch.ones_like(x1)
            x_zero = torch.zeros_like(x1)
            self.linrdp = torch.where(x2 <= 0., x_zero, x_one)

        x3 = self.linear2(x2) # Readout 2
        x4 = x3
        
        with torch.no_grad():
            x_one = torch.ones_like(x3)
            x_zero = torch.zeros_like(x3)
            self.linrdp2 = torch.where(x4 <= 0., x_zero, x_one)
        
        return F.log_softmax(x4, dim=1)