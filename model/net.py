# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/21 15:37
# @Author:Ma Jie
# @FileName: net.py
# -----------------------------------------------
import torch
import torch.nn as nn
from model.gat import GAT
from model.layer import MultiHeadAttentionLayer, AttFlat
from utils.util import make_mask, make_mask_opt_num
import numpy as np

np.set_printoptions(threshold=1e6)


class TextualNet(nn.Module):
    def __init__(self, cfg):
        super(TextualNet, self).__init__()
        # [batch_size, max_option_count, max_nodes, gat_emb] = [b, 7, 70, 300]
        self.text_gat = GAT(cfg.gat_node_emb, cfg.gat_hid, cfg.gat_node_emb, cfg.gat_dropout, cfg.gat_alpha,
                            cfg.gat_heads)
        # [batch_size, max_option_count, 1, hidden_states] = [b, 7, 1, 300], questions and options are input to lstm respectively.
        self.lstm = nn.LSTM(
            input_size=cfg.word_emb,
            hidden_size=cfg.lstm_hid,
            num_layers=cfg.lstm_layer,
            batch_first=True,
        )
        # [batch_size, max_option_count, max_nodes, gat_emb] = [b, 7, 70, 300]
        self.att = MultiHeadAttentionLayer(
            hid_dim=cfg.multi_hid,
            n_heads=cfg.multi_heads,
            dropout=cfg.multi_dropout,
            device='cuda'
        )
        # [max_opt_count, max_gat_nodes] = [1, 70]
        self.W = nn.Parameter(torch.zeros(size=(1, cfg.gat_max_nodes)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(cfg.gat_alpha)
        self.flat = AttFlat(cfg)
        # [batch_size, max_opt_count, 2 * word_emb, 1]
        self.classify = nn.Linear(3 * cfg.word_emb, 1)

    def forward(self, que_emb, opt_emb, adjacency_matrices, node_emb, cfg):
        batch_size = que_emb.shape[0]
        que_mask = make_mask(que_emb)
        que_emb, _ = self.lstm(que_emb)
        que_emb = que_emb.repeat(1, 1, cfg.max_opt_count, 1)
        que_emb = torch.reshape(que_emb, (batch_size, cfg.max_opt_count, cfg.max_que_len, cfg.word_emb))

        opt_mask = make_mask(opt_emb)
        opt_mask_num = make_mask_opt_num(opt_emb)
        opt_emb = torch.reshape(opt_emb, (-1, cfg.max_opt_len, cfg.word_emb))
        opt_emb, _ = self.lstm(opt_emb)
        opt_emb = torch.reshape(opt_emb, (batch_size, cfg.max_opt_count, cfg.max_opt_len, cfg.word_emb))

        gat_node_emb = self.text_gat(node_emb, adjacency_matrices, cfg)
        # [batch_size, max_opt, max_nodes, gat_node_emb]
        gat_node_emb = self.att(gat_node_emb, que_emb, que_emb, cfg)
        # [batch_size, max_opt, 1, gat_node_emb]
        graph_emb = self.leakyrelu(torch.matmul(self.W, gat_node_emb))
        graph_emb = graph_emb.squeeze(2)
        que_mask = torch.reshape(que_mask.repeat(1, cfg.max_opt_count),
                                 (batch_size, cfg.max_opt_count, cfg.max_que_len))
        que_emb = self.flat(que_emb, que_mask, cfg)
        opt_emb = self.flat(opt_emb, opt_mask, cfg)
        fusion_feat = torch.cat((que_emb, opt_emb, graph_emb), -1)
        proj_feat = self.classify(fusion_feat)
        proj_feat = torch.reshape(proj_feat, (batch_size, 1, cfg.max_opt_count))
        opt_mask_num = opt_mask_num.unsqueeze(1)
        proj_feat = proj_feat.masked_fill(opt_mask_num == 1, -1e10)
        return proj_feat


class DiagramNet(nn.Module):
    def __init__(self, gfeat, dfeat, ghid, gout, gdropout, galpha, gheads):
        super(DiagramNet, self).__init__()

        self.text_gat = GAT(gfeat, ghid, gout, gdropout, galpha, gheads)
        self.dia_gat = GAT(dfeat, ghid, gout, gdropout, galpha, gheads)
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=300,
            num_layers=1,
            batch_first=True
        )

    def forward(self, *input):
        pass
