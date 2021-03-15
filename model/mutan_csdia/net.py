# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2020/11/1 17:19
# @Author:wyx
# @File : net.py
# -----------------------------------------------
import torch
import torch.nn as nn

from model.mutan_csdia.layer import MuTAN, FlattenAtt, SimCLR
from torch.nn.utils.weight_norm import weight_norm
from utils.util import make_mask


class Net(nn.Module):
    def __init__(self, cfgs):
        """
        :param cfgs: configurations of XTQA.
        :param pretrained_emb:
        :param token_size: the size of vocabulary table.
        """
        super(Net, self).__init__()
        # use simclr to encode the diagram
        self.simclr = SimCLR(cfgs)
        self.flat = FlattenAtt(cfgs)

        # bilinear attention networks
        self.backbone = MuTAN(v_relation_dim=cfgs.relation_dim, num_hid=cfgs.hidden_size,
                              gamma=cfgs.mutan_gamma)

        # Classification layers
        layers = [
            weight_norm(nn.Linear(4864, cfgs.flat_out_size), dim=None),
            nn.ReLU(),
            nn.Dropout(cfgs.classifer_dropout_r, inplace=True),
            weight_norm(nn.Linear(cfgs.flat_out_size, 1), dim=None)
        ]
        self.classifer = nn.Sequential(*layers)

    def forward(self, que_emb, dia_f, opt_emb, dia_matrix, dia_node_emb, cfgs):
        """
        :param que_ix: the index of questions
        :param opt_ix: the index of options
        :param cp_ix: the closest paragraph that is extracted by TF-IDF method
        """
        batch_size = que_emb.shape[0]

        dia_feat = self.simclr(dia_f)
        dia_feat = dia_feat.reshape(batch_size, -1)
        dia_feat = dia_feat.unsqueeze(1)

        fusion_feat = self.backbone(dia_feat, que_emb.squeeze(1))  # [8,4096]  BAN得到的是[b,15,1024]

        # fusion_feat = self.flatten(fusion_feat.sum(1))
        fusion_feat = fusion_feat.repeat(1, cfgs.opt_num).reshape(batch_size, cfgs.opt_num, -1)
        fuse_opt_feat = torch.cat((fusion_feat, opt_emb.squeeze(2)), dim=-1)  # (8,4,4096)(8,4,1024)
        proj_feat = self.classifer(fuse_opt_feat).squeeze(-1)

        return proj_feat
