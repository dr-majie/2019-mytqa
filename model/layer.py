# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/21 15:27
# @Author:Jie Ma
# @FileName: layer.py
# -----------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

np.set_printoptions(threshold=1e9)


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, cfg):
        super(MHAtt, self).__init__()
        self.cfg = cfg

        self.linear_v = nn.Linear(cfg.multi_hidden, cfg.multi_hidden)
        self.linear_k = nn.Linear(cfg.multi_hidden, cfg.multi_hidden)
        self.linear_q = nn.Linear(cfg.multi_hidden, cfg.multi_hidden)
        self.linear_merge = nn.Linear(cfg.multi_hidden, cfg.multi_hidden)

        self.dropout = nn.Dropout(cfg.multi_drop_out)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.cfg.multi_heads,
            int(self.cfg.multi_hidden / self.cfg.multi_heads)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.cfg.multi_heads,
            int(self.cfg.multi_hidden / self.cfg.multi_heads)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.cfg.multi_heads,
            int(self.cfg.multi_hidden / self.cfg.multi_heads)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.cfg.multi_hidden
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
            # print(scores.detach().cpu().numpy()[0][0])
        att_map = F.softmax(scores, dim=-1)
        # print(att_map.detach().cpu().numpy()[0][0])
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, cfg):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=cfg.multi_hidden,
            mid_size=cfg.mlp_hid,
            out_size=cfg.multi_hidden,
            multi_drop_out=cfg.multi_drop_out,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, cfg):
        super(SA, self).__init__()

        self.mhatt = MHAtt(cfg)
        self.dropout1 = nn.Dropout(cfg.multi_drop_out)
        self.norm1 = LayerNorm(cfg.multi_hidden)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        return y


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlatText(nn.Module):
    def __init__(self, cfg):
        super(AttFlatText, self).__init__()

        self.mlp = MLP(
            in_size=cfg.mlp_in,
            mid_size=cfg.mlp_hid,
            out_size=cfg.glimpse,
            multi_drop_out=cfg.mlp_dropout,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            cfg.mlp_in * cfg.glimpse,
            cfg.mlp_out,
            bias=False
        )

    def forward(self, x, x_mask, cfg):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.unsqueeze(-1) == 1,
            -9e15
        )
        att = F.softmax(att, dim=2)

        att_list = []
        for i in range(cfg.glimpse):
            att_list.append(
                torch.sum(att[:, :, :, i: i + 1] * x, dim=2)
            )

        x_atted = torch.cat(att_list, dim=2)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# ------------------------------
# ---- Flatten the diagram ----
# ------------------------------

class AttFlatDiagram(nn.Module):
    def __init__(self, cfg):
        super(AttFlatDiagram, self).__init__()

        self.mlp = MLP(
            in_size=cfg.mlp_in,
            mid_size=cfg.mlp_hid,
            out_size=cfg.glimpse,
            multi_drop_out=cfg.mlp_dropout,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            cfg.mlp_in * cfg.glimpse,
            cfg.mlp_out,
            bias=False
        )

    def forward(self, x, x_mask, cfg):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.unsqueeze(-1) == 1,
            -9e15
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(cfg.glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, multi_drop_out=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, multi_drop_out=multi_drop_out, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size, bias=False)

    def forward(self, x):
        return self.linear(self.fc(x))


class FC(nn.Module):
    def __init__(self, in_size, out_size, multi_drop_out=0., use_relu=True):
        super(FC, self).__init__()
        self.multi_drop_out = multi_drop_out
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size, bias=False)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if multi_drop_out > 0:
            self.dropout = nn.Dropout(multi_drop_out)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.multi_drop_out > 0:
            x = self.dropout(x)

        return x


class MultiSA(nn.Module):
    def __init__(self, cfg):
        super(MultiSA, self).__init__()

        self.enc_list = nn.ModuleList([SA(cfg) for _ in range(cfg.sa_layer)])

    def forward(self, x, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.leakyrelu = nn.ReLU(inplace=True)

    def forward(self, input, adj, cfg):
        h = torch.matmul(input, self.W)
        N = cfg.max_diagram_node
        batch_size = h.shape[0]
        a_input = torch.cat(
            [h.repeat(1, 1, N).view(batch_size, N * N, -1), h.repeat(1, N, 1)],
            dim=-1).view(batch_size, N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = torch.where(adj > 0, attention, torch.zeros_like(attention))
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


# ----------------------------------
# ---- Inter-Modality Attention ----
# ----------------------------------
class IMA(nn.Module):
    def __init__(self, cfg):
        super(IMA, self).__init__()

        self.mhatt = MHAtt(cfg)
        # self.ffn = FFN(cfg)

        self.dropout1 = nn.Dropout(cfg.multi_drop_out)
        self.norm1 = LayerNorm(cfg.multi_hidden)

        # self.dropout2 = nn.Dropout(cfg.multi_drop_out)
        # self.norm2 = LayerNorm(cfg.multi_hidden)

    def forward(self, q, i, q_mask, i_mask):
        i = self.norm1(i + self.dropout1(
            self.mhatt(v=q, k=q, q=i, mask=q_mask)
        ))

        # i = self.norm2(i + self.dropout2(
        #     self.ffn(i)
        # ))

        return i


# -----------------------
# ---- Intra_2_inter ----
# -----------------------
class INTRA_2_INTER(nn.Module):
    def __init__(self, cfg):
        super(INTRA_2_INTER, self).__init__()
        self.inter_att_list = nn.ModuleList(IMA(cfg) for _ in range(cfg.intra2inter_layer))

    def forward(self, que, diagram, que_mask, diagram_mask):
        for inter_att in self.inter_att_list:
            que_update = inter_att(diagram, que, diagram_mask, que_mask)
            diagram_update = inter_att(que, diagram, que_mask, diagram_mask)

            que = que_update
            diagram = diagram_update

        return que, diagram
