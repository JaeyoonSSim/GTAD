import torch.nn as nn
import torch.nn.functional as F
import sys


class MLP(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(self.__class__, self).__init__()
        """
        self.pred = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.LeakyReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.LeakyReLU(),
            nn.Linear(hid_feats, out_feats),
        )
        """
        self.pred = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.LeakyReLU(),
            nn.Linear(hid_feats, out_feats),
        )

    def forward(self, x): # (sample, node, feature)
        x = x.reshape(x.shape[0],-1)
        x = self.pred(x)

        return F.log_softmax(x, dim=1)
