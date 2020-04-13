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


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, cfg):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, cfg) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj, cfg))
        # return F.log_softmax(x, dim=1)
        # return the representation of each node
        # x = torch.cat([att(x, adj, cfg) for att in self.attentions], dim=-1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.reshape(x, (-1, batch_size, cfg.max_opt_count, cfg.gat_max_nodes, cfg.gat_hid))
        # x = torch.mean(x, 0)
        # x = F.elu(x)
        return x
