# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/21 15:37
# @Author:Ma Jie
# @FileName: gat.py
# -----------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rafr_tqa.layer import GraphAttentionLayer

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj, cfg):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, cfg) for att in self.attentions], dim=-1)

        return x
