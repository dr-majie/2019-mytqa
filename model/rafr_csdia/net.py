# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/21 15:37
# @Author:Ma Jie
# @FileName: net.py
# -----------------------------------------------
import torch
import torch.nn as nn
from model.rafr_csdia.gat import GAT
from utils.util import make_mask, make_mask_num
from model.rafr_csdia.layer import LayerNorm, AttFlatDiagram, INTRA_2_INTER
import numpy as np

np.set_printoptions(threshold=1e6)


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.gat = GAT(
            cfg.init_word_emb,
            cfg.gat_hidden,
            cfg.gat_dropout,
            cfg.gat_alpha,
            cfg.gat_heads
        )
        self.intra2inter = INTRA_2_INTER(cfg)

        self.flat = AttFlatDiagram(cfg)
        self.ln = LayerNorm(cfg.mlp_out * 6)

        self.classify = nn.Linear(cfg.mlp_out * 6, 1)

    def forward(self, que_emb, opt_emb, dia_matrix, dia_node_emb, cfg):
        batch_size = que_emb.shape[0]

        que_mask = make_mask(que_emb)
        dia_node_mask = make_mask(dia_node_emb)
        dia_node_feat = self.gat(dia_node_emb, dia_matrix, cfg)
        dia_node_feat = dia_node_feat.masked_fill(dia_node_mask.unsqueeze(-1), 0.)

        que_feat, dia_node_feat = self.intra2inter \
            (que_emb,
             dia_node_feat,
             que_mask.unsqueeze(1).unsqueeze(2),
             dia_node_mask.unsqueeze(1).unsqueeze(2))

        dia_feat = self.flat(dia_node_feat, dia_node_mask, cfg)
        dia_feat = dia_feat.repeat(1, cfg.opt_num).reshape(-1, cfg.opt_num, cfg.mlp_out)

        que_feat = que_feat.squeeze(1).repeat(1, cfg.opt_num).reshape(batch_size, cfg.opt_num, -1)
        opt_feat = opt_emb.squeeze(2)

        sim_q_o = que_feat * opt_feat
        sim_d_o = dia_feat * opt_feat
        sim_d_o_q = que_feat * opt_feat * dia_feat

        fusion_feat = torch.cat((que_feat, dia_feat, opt_feat, sim_d_o, sim_q_o, sim_d_o_q), dim=-1)
        fusion_feat = self.ln(fusion_feat)
        scores = self.classify(fusion_feat).squeeze(-1)
        return scores
