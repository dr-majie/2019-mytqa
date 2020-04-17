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
            dropout_r=cfg.multi_drop_out,
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

class AttFlat(nn.Module):
    def __init__(self, cfg):
        super(AttFlat, self).__init__()

        self.mlp = MLP(
            in_size=cfg.mlp_in,
            mid_size=cfg.mlp_hid,
            out_size=cfg.glimpse,
            dropout_r=cfg.mlp_dropout,
            use_relu=True
        )
        # self.mlp = MLP_new(
        #     cfg.word_emb,
        #     cfg.glimpse,
        #     dropout_r=cfg.mlp_dropout,
        #     use_relu=True
        # )

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


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size, bias=False)

    def forward(self, x):
        return self.linear(self.fc(x))


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size, bias=False)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
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