# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2020/11/17 19:22
# @Author:wyx
# @File : net.py
# -----------------------------------------------

import torch
import torch.nn as nn
from utils.util import make_mask
from model.mfb_csdia.layer import CoAtt, SimCLR
from torch.nn.utils.weight_norm import weight_norm


# -------------------------------------------------------
# ---- Main MFB/MFH model with Co-Attention Learning ----
# -------------------------------------------------------
class Net(nn.Module):
    def __init__(self, cfgs):
        super(Net, self).__init__()
        self.cfgs = cfgs

        self.dropout = nn.Dropout(cfgs.dropout_r)
        self.dropout_lstm = nn.Dropout(cfgs.dropout_r)

        self.simclr = SimCLR(cfgs)
        self.backbone = CoAtt(cfgs)

        if cfgs.high_order:  # MFH
            self.classifer = nn.Linear(2 * cfgs.mfb_o + cfgs.lstm_out_size, 1)
        else:  # MFB
            self.classifer = nn.Linear(cfgs.mfb_o + cfgs.lstm_out_size, 1)

    def forward(self, que_emb, dia_f, opt_emb, dia_matrix, dia_node_emb, cfg):

        batch_size = que_emb.shape[0]

        dia_feat = self.simclr(dia_f)
        dia_feat = dia_feat.reshape(batch_size, -1)
        dia_feat = dia_feat.unsqueeze(1)

        fusion_feat = self.backbone(dia_feat, que_emb)  # MFH:(N, 2*O) / MFB:(N, O)
        # proj_feat = self.proj(z)                # (N, answer_size)
        fusion_feat = fusion_feat.repeat(1, self.cfgs.opt_num).reshape(batch_size, self.cfgs.opt_num, -1)  # (8,4,1024)
        fuse_opt_feat = torch.cat((fusion_feat, opt_emb.squeeze(2)), dim=-1)  # (8,4,4096)(8,4,1024)
        proj_feat = self.classifer(fuse_opt_feat).squeeze(-1)

        return proj_feat
