import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
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

class GraphConvolution(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GraphConvolution, self).__init__()
        self.gc1 = GraphConvolutionLayer(in_dim, hid_dim) 
        self.gc2 = GraphConvolutionLayer(hid_dim, out_dim) 
        self.dropout = dropout

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, adj): # Graph convolution part 
        out = self.gc1.forward(x, adj)
        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.gc2.forward(out, adj)
        out = F.relu(out)

        return out

class Projector(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(Projector, self).__init__()
        self.proj_lin1 = nn.Linear(in_dim, hid_dim)
        self.proj_lin2 = nn.Linear(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x): # (sample, node, feature)
        out = x.reshape(x.shape[0],-1)
        out = self.proj_lin1(out)
        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.proj_lin2(out)
        
        return out

class Classifier(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(Classifier, self).__init__()
        self.lin1 = nn.Linear(adj_dim * hid_dim * in_dim, adj_dim * hid_dim * in_dim // 2) # Readout
        self.lin2 = nn.Linear(adj_dim * hid_dim * in_dim // 2, out_dim) # Predictor
        self.dropout = dropout
        
    def forward(self, x):
        out = x.reshape(x.shape[0], -1)
        out = self.lin1(out)
        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.lin2(out)
        out = F.log_softmax(out, dim=1)
        
        return out 

class OURS(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, ratio, dropout):
        super(OURS, self).__init__()
        self.dropout = dropout       
        self.n_pts = adj_dim
        self.n_feats = in_dim
        self.hid_dim = hid_dim
        self.n_layers = 1
        self.init_scale = 3
        self.ratio = ratio
        
        self.scale_per_feat_ = []
        self.conv_layer_per_feat_ = []
        for i in range(in_dim):
            self.scale_per_feat_.append(nn.Parameter(torch.empty((self.n_pts)).fill_(self.init_scale), requires_grad=True))
            
            # self.conv_layer_per_feat_.append(LinearEncoder(adj_dim, 3, hid_dim, hid_dim, dropout)) # 1
            # self.conv_layer_per_feat_.append(LinearEncoder(adj_dim, 1, hid_dim, hid_dim, dropout)) # 2
            # self.conv_layer_per_feat_.append(GraphConvolution(3, hid_dim, hid_dim, dropout)) # 3, 5
            self.conv_layer_per_feat_.append(GraphConvolution(1, hid_dim, hid_dim, dropout)) # 4, Ours
            
        self.scale_per_feat = nn.ParameterList(self.scale_per_feat_)
        self.conv_layer_per_feat = nn.ModuleList(self.conv_layer_per_feat_)
        
        self.classifier = Classifier(adj_dim, in_dim, hid_dim, out_dim, dropout)

        c = copy.deepcopy
        attn = MultiHeadedAttention(in_dim, in_dim * hid_dim)
        fd_fwd = FeedForward(adj_dim, in_dim, hid_dim, hid_dim, dropout)
        self.attn_layers_ = []
        for i in range(self.n_layers):
            self.attn_layers_.append(GraAttenLayer(self.n_feats * hid_dim, c(attn), c(fd_fwd), dropout=0.5))
        self.attn_layers = nn.ModuleList(self.attn_layers_)
    
    def compute_hk(self, eigval, eigvec, t):
        n_samples = eigval.shape[0]
        eigval = eigval.type(torch.float)
        eigvec = eigvec.type(torch.float)
        
        one = torch.ones_like(eigvec)
        ftr = torch.mul(one, torch.exp(-eigval).reshape(n_samples, 1, -1)) ** t.reshape(-1, 1) 
        
        hk = torch.matmul(torch.mul(eigvec, ftr), eigvec.transpose(-1,-2)) 
        hk[hk < 0.1] = 0

        return hk
    
    def forward(self, x, eigenvalue, eigenvector, device):
        HKs = []
        emb_per_feat = []
        for i in range(self.n_feats):
            # self.scale_per_feat[i].data.clamp_(1e-5, 160)
            HKs.append(self.compute_hk(eigenvalue, eigenvector, self.scale_per_feat[i].to(device)))
            emb_per_feat.append(self.conv_layer_per_feat[i](x[:,:,i].unsqueeze(-1), HKs[i]))
        
        out = torch.cat(emb_per_feat, dim=-1)

        for n in range(self.n_layers):
            out = self.attn_layers[n](out)
        
        self.attn_score = torch.mean(self.attn_layers[-1].attn_score, dim=0)
        out = self.classifier(out)
        
        return out
    
    def inference_abl(self, x, adjacency, eigenvalue, eigenvector, device):
        HKs = []
        emb_per_feat = []
        
        """ 1 """
        # for i in range(self.n_feats):
        #     emb_per_feat.append(self.conv_layer_per_feat[i](x, adjacency))
        """ 2 """
        for i in range(self.n_feats):
            emb_per_feat.append(self.conv_layer_per_feat[i](x[:,:,i].unsqueeze(-1), adjacency))
        """ 3 """
        # for i in range(self.n_feats):
        #     emb_per_feat.append(self.conv_layer_per_feat[i](x, adjacency))
        """ 4 """
        # for i in range(self.n_feats):
        #     emb_per_feat.append(self.conv_layer_per_feat[i](x[:,:,i].unsqueeze(-1), adjacency))
        """ 5 """
        # for i in range(self.n_feats):
        #     HKs.append(self.compute_hk(eigenvalue, eigenvector, self.scale_per_feat[i].to(device)))
        #     emb_per_feat.append(self.conv_layer_per_feat[i](x, HKs[i]))
        
        out = torch.cat(emb_per_feat, dim=-1)

        for n in range(self.n_layers):
            out = self.attn_layers[n](out)
        
        self.attn_score = torch.mean(self.attn_layers[-1].attn_score, dim=0)
        out = self.classifier(out)
        
        return out
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features=layer.size=512
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class GraAttenLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GraAttenLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        module_list = []
        for i in range(2):
            module_list.append(SublayerConnection(size, dropout))
        self.sublayer = nn.ModuleList(module_list)
        self.size = size

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x)) # Multi-Head Attention
        self.attn_score = self.self_attn.attn_score ##
        return self.sublayer[1](x, self.feed_forward) # Feed-Forward Network


def attention(Q, K, V, dropout=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    attn_score = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn_score = dropout(attn_score)
    
    self_attn_value = torch.matmul(attn_score, V)
    
    return self_attn_value, attn_score


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_feats, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_feats == 0
        self.d_k = d_model // n_feats
        self.h = n_feats

        self.linears_ = []
        for i in range(4):
            self.linears_.append(nn.Linear(d_model, d_model))
        self.linears = nn.ModuleList(self.linears_)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        n_samples = query.size(0)
        
        Q, K, V = [l(x).view(n_samples, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]
        
        self_attn_value, self.attn_score = attention(Q, K, V, dropout=self.dropout)
        
        out = self_attn_value.transpose(1, 2).contiguous().view(n_samples, -1, self.h * self.d_k)
        
        return self.linears[-1](out)
    
class FeedForward(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(FeedForward, self).__init__()
        self.lin1 = nn.Linear(hid_dim * in_dim, hid_dim * in_dim) # Readout
        self.lin2 = nn.Linear(hid_dim * in_dim, out_dim * in_dim) # Predictor
        self.dropout = dropout
        
    def forward(self, x):
        out = self.lin1(x)
        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.lin2(out)

        return out 

class LinearEncoder(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(LinearEncoder, self).__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim) # Readout
        self.lin2 = nn.Linear(hid_dim, hid_dim) # Predictor
        self.dropout = dropout
        
    def forward(self, x, adj):
        out = self.lin1(x)
        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.lin2(out)

        return out 