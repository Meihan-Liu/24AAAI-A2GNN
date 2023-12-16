import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear

from layers import GCNConv, GRL, DomainDiscriminator


class Model(torch.nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.cls = GCNConv(self.nhid, self.num_classes)
        self.domain_discriminator = Linear(self.nhid, 2)
        self.grl = GRL()

    def forward(self, x, edge_index, conv_time=30):
        x = self.feat_bottleneck(x, edge_index, conv_time)
        x = self.feat_classifier(x, edge_index)

        return x
    
    def feat_bottleneck(self, x, edge_index, conv_time=30):
        x = self.conv1(x, edge_index, 0)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, conv_time)
        x = F.relu(x)

        return x
    
    def feat_classifier(self, x, edge_index, conv_time=1):
        x = self.cls(x, edge_index, conv_time)
        
        return x
    
    def domain_classifier(self, x, edge_index=None, conv_time=1):
        h_grl = self.grl(x)
        d_logit = self.domain_discriminator(h_grl)
        
        return d_logit