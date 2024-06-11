import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor) -> torch.tensor:
        pass

class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)
        # self.cnn2 = torch.nn.Conv2d(in_planes, planes, (1,self.d), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)


class BrainNetCNN(BaseModel):
    def __init__(self, adj_dim, nfeat, hid_dim, out_dim):
        super().__init__()
        self.in_planes = 1
        self.d = adj_dim

        self.e2econv1 = E2EBlock(1, hid_dim, adj_dim, bias=True)
        self.e2econv2 = E2EBlock(hid_dim, hid_dim*2, adj_dim, bias=True)
        self.E2N = torch.nn.Conv2d(hid_dim*2, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, hid_dim*4, (self.d, 1))
        self.dense1 = torch.nn.Linear(hid_dim*4, hid_dim*2)
        self.dense2 = torch.nn.Linear(hid_dim*2, hid_dim)
        self.dense3 = torch.nn.Linear(hid_dim, out_dim)

    def forward(self,
                node_feature: torch.tensor,
                edge_feature: torch.tensor):
        # out = node_feature.transpose(-1,-2)
        # out = torch.diag_embed(out)
        out = edge_feature.unsqueeze(dim=1)
        out = F.leaky_relu(self.e2econv1(out), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(
            self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(
            self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(
            self.dense2(out), negative_slope=0.33), p=0.5)
        out = self.dense3(out)
        # out = F.leaky_relu(self.dense3(out), negative_slope=0.33)
        return F.log_softmax(out, dim=1)