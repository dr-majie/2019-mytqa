# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/8/6 15:30
# @Author:Ma Jie
# -----------------------------------------------
import torch
import torch.nn as nn
from model.ban_csdia.layer import SimCLR, BAN
from torch.nn.utils.weight_norm import weight_norm


class Net(nn.Module):
    def __init__(self, cfgs):
        """
        :param cfgs: configurations of XTQA.
        :param pretrained_emb:
        :param token_size: the size of vocabulary table.
        """
        super(Net, self).__init__()
        self.cfgs = cfgs

        # use simclr to encode the diagram
        self.simclr = SimCLR(cfgs)

        # bilinear attention networks
        self.backbone = BAN(cfgs)

        # Classification layers
        layers = [
            weight_norm(nn.Linear(cfgs.hidden_size, cfgs.flat_out_size), dim=None),
            nn.ReLU(),
            nn.Dropout(cfgs.classifer_dropout_r, inplace=True),
            # weight_norm(nn.Linear(cfgs.flat_out_size, 1), dim=None)
        ]
        self.flatten = nn.Sequential(*layers)
        # self.classifer = nn.CosineSimilarity(dim=2)
        layers_classifer = [
            weight_norm(nn.Linear(cfgs.hidden_size * 2, cfgs.flat_out_size), dim=None),
            nn.ReLU(),
            nn.Dropout(cfgs.classifer_dropout_r, inplace=True),
            weight_norm(nn.Linear(cfgs.flat_out_size, 1), dim=None)
        ]
        self.classifer = nn.Sequential(*layers_classifer)

    def forward(self, que_emb, dia_f, opt_emb, dia_matrix, dia_node_emb, cfg):
        """
        :param que_ix: the index of questions
        :param opt_ix: the index of options
        :param dia: the diagram corresponding the above question
        :param ins_dia: the instructional diagram corresponding to the lesson that contains the above question
        :param cp_ix: the closest paragraph that is extracted by TF-IDF method
        """
        batch_size = que_emb.shape[0]

        dia_feat = self.simclr(dia_f)
        dia_feat = dia_feat.reshape(batch_size, -1)

        dia_feat = dia_feat.unsqueeze(1)
        fusion_feat = self.backbone(que_emb, dia_feat)

        fusion_feat = self.flatten(fusion_feat.sum(1))
        fusion_feat = fusion_feat.repeat(1, self.cfgs.opt_num).reshape(batch_size, self.cfgs.opt_num, -1)
        opt_feat = opt_emb.squeeze(2)
        fuse_feat = torch.cat((fusion_feat, opt_feat), dim=-1)
        proj_feat = self.classifer(fuse_feat).squeeze(-1)
        return proj_feat
