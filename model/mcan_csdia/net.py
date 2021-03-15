# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/11/6 15:30
# @Author:Ma Jie
# -----------------------------------------------


import torch
import torch.nn as nn
from model.mcan_csdia.layer import MCA_ED, SimCLR, AttFlat, LayerNorm


class Net(nn.Module):
    """
        :param cfgs: configurations of XTQA.
        :param pretrained_emb:
        :param token_size: the size of vocabulary table.
    """

    def __init__(self, cfgs):
        super(Net, self).__init__()
        self.cfgs = cfgs
        # use simclr to encode features of diagrams
        self.simclr = SimCLR(cfgs)
        self.fc_q = nn.Linear(768, cfgs.hidden_size)
        self.fc_d = nn.Linear(cfgs.dia_feat_size, cfgs.hidden_size)
        self.backbone = MCA_ED(cfgs)
        self.attflat_dia = AttFlat(cfgs)
        self.attflat_lang = AttFlat(cfgs)

        self.proj_norm = LayerNorm(cfgs.flat_out_size)
        self.classifer = nn.Linear(cfgs.flat_out_size + 768, 1)

    def forward(self, que_emb, dia_f, opt_emb, dia_matrix, dia_node_emb, cfg):
        """
        :param que_ix: the index of questions
        :param opt_ix: the index of options
        :param dia: the diagram corresponding the above question
        :param ins_dia: the instructional diagram corresponding to the lesson that contains the above question
        :param cp_ix: the closest paragraph that is extracted by TF-IDF method
        """
        batch_size = que_emb.shape[0]

        que_feat = self.fc_q(que_emb)
        lang_feat_mask = self.make_mask(que_feat)

        dia_feat = self.simclr(dia_f)
        dia_feat = self.fc_d(dia_feat)
        dia_feat = dia_feat.reshape(batch_size, -1)
        dia_feat = dia_feat.unsqueeze(1)

        dia_feat_mask = self.make_mask(dia_feat)
        # Backbone Framework
        lang_feat, dia_feat = self.backbone(
            que_feat,
            dia_feat,
            lang_feat_mask,
            dia_feat_mask
        )
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        dia_feat = self.attflat_dia(
            dia_feat,
            dia_feat_mask
        )

        fusion_feat = lang_feat + dia_feat
        fusion_feat = self.proj_norm(fusion_feat)
        fusion_feat = fusion_feat.repeat(1, self.cfgs.opt_num).reshape(batch_size, self.cfgs.opt_num,
                                                                       self.cfgs.flat_out_size)

        fuse_feat = torch.cat((fusion_feat, opt_emb.squeeze(2)), dim=-1)
        proj_feat = self.classifer(fuse_feat).squeeze(-1)

        return proj_feat

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
