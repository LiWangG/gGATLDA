import torch
import math
import torch.nn as nn
from torch_geometric.nn import MessagePassing,GATConv
import torch.nn.functional as F
from torch.nn import Linear, Conv1d,AdaptiveMaxPool2d
from torch_geometric.nn import GCNConv, RGCNConv, global_sort_pool, global_add_pool, global_max_pool,global_mean_pool
from torch_geometric.utils import dropout_adj
from util_functions import *
import pdb
import time
from torch.autograd import Variable
from ranger import Ranger
from ranger import RangerVA
from ranger import RangerQH



class gGATLDA(torch.nn.Module):
    # The gGATLDA model use GCN layer + GAT layer
    def __init__(self, in_features, gconv=GATConv, latent_dim=[16, 16, 16, 16], side_features=False, n_side_features=0):
        super(gGATLDA, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.conv1=GCNConv((in_features), 16)
        self.convs.append(gconv(16, latent_dim[0],heads=8,dropout=0.2))
        self.convs.append(gconv(16*8, 16,heads=8,dropout=0.2))
        self.convs.append(gconv(16*8, 16,heads=8,dropout=0.2))
        self.convs.append(gconv(16*8, 2,heads=1,dropout=0.2))
        self.conv2=gconv(in_channels=2*8,out_channels=2,dropout=0.2,heads=1,concat=True)
        self.lin1 = Linear(3*sum(latent_dim), 8)
        self.lin2 = Linear(2*4*16, 8)

    def forward(self, data):
        start = time.time()
        x, edge_index, batch = data.x, data.edge_index, data.batch
        concat_states=[]
        x = F.elu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        x = torch.cat([x[users], x[items]], 1)
        return F.log_softmax(x, dim=1)

    def predict(self,data):
        out=self.forward(data)
        return out







