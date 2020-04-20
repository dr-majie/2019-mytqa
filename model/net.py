# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/21 15:37
# @Author:Ma Jie
# @FileName: net.py
# -----------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gat import GAT
from utils.util import make_mask, make_mask_num
from model.layer import AttFlat, LayerNorm, MultiSA, INTRA_2_INTER
import numpy as np
import math

np.set_printoptions(threshold=1e6)


class TextualNetBeta(nn.Module):
    def __init__(self, cfg):
        super(TextualNetBeta, self).__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.init_word_emb,
            hidden_size=cfg.lstm_hid,
            num_layers=cfg.lstm_layer,
            batch_first=True,
        )
        self.att = MultiSA(cfg)
        self.flat = AttFlat(cfg)

        self.ln = LayerNorm(cfg.mlp_out)
        self.classify = nn.Linear(cfg.mlp_out * 4, 1)

    def forward(self, que_emb, opt_emb, closest_sent_emb, cfg):
        batch_size = que_emb.shape[0]
        que_mask = make_mask(que_emb)
        que_feat, _ = self.lstm(que_emb)
        que_feat = que_feat.masked_fill(que_mask.unsqueeze(-1) == 1, 0.)
        que_feat = self.att(que_feat, que_mask.unsqueeze(1).unsqueeze(2))
        que_feat = que_feat.masked_fill(que_mask.unsqueeze(-1) == 1, 0.)

        opt_mask = make_mask(opt_emb)
        opt_sum = make_mask_num(opt_emb)
        opt_feat = opt_emb.reshape(-1, cfg.max_opt_len, cfg.init_word_emb)
        opt_feat, _ = self.lstm(opt_feat)
        opt_feat = opt_feat.masked_fill(opt_mask.reshape(-1, cfg.max_opt_len).unsqueeze(-1) == 1, 0.)
        opt_feat = self.att(opt_feat, opt_mask.reshape(batch_size * cfg.max_opt_count, -1).unsqueeze(1).unsqueeze(2))
        opt_feat = opt_feat.reshape(batch_size, cfg.max_opt_count, cfg.max_opt_len, -1)
        opt_feat = opt_feat.masked_fill(opt_mask.unsqueeze(-1) == 1, 0.)

        closest_sent_mask = make_mask(closest_sent_emb)
        closest_sent_sum = make_mask_num(closest_sent_emb)
        closest_sent_feat = closest_sent_emb.reshape(-1, cfg.max_words_sent, cfg.init_word_emb)
        closest_sent_feat, _ = self.lstm(closest_sent_feat)
        closest_sent_feat = closest_sent_feat.masked_fill(
            closest_sent_mask.reshape(-1, cfg.max_words_sent).unsqueeze(-1) == 1, 0.)
        closest_sent_feat = self.att(closest_sent_feat,
                                     closest_sent_mask.reshape(-1, cfg.max_words_sent).unsqueeze(1).unsqueeze(2))
        closest_sent_feat = closest_sent_feat.reshape(batch_size, cfg.max_sent_para, cfg.max_words_sent, -1)
        closest_sent_feat = closest_sent_feat.masked_fill(closest_sent_mask.unsqueeze(-1) == 1, 0.)

        que_csf, que_csf_mask, opt_csf, opt_csf_mask = self.find_context(que_feat,
                                                                         que_mask,
                                                                         opt_feat,
                                                                         opt_mask,
                                                                         opt_sum,
                                                                         closest_sent_feat,
                                                                         closest_sent_mask,
                                                                         cfg)

        que_feat = que_feat.repeat(1, cfg.max_opt_count, 1).reshape(batch_size, cfg.max_opt_count, -1, cfg.word_emb)
        que_mask = que_mask.repeat(1, cfg.max_opt_count).reshape(batch_size, cfg.max_opt_count, -1)

        flat_que = self.flat(que_feat, que_mask, cfg)
        flat_opt = self.flat(opt_feat, opt_mask, cfg)
        flat_que_csf = self.flat(que_csf, que_csf_mask, cfg)
        flat_opt_csf = self.flat(opt_csf, opt_csf_mask, cfg)

        # fusion_feat = torch.cat((flat_que, flat_opt, flat_que_csf, flat_opt_csf), dim=-1)
        fusion_feat = torch.cat((flat_que, flat_que_csf, flat_opt, flat_opt_csf), dim=-1)
        scores = self.classify(fusion_feat).squeeze(-1)
        scores = scores.masked_fill(opt_sum == 1, -9e15)
        return scores

    def find_context(self, que_feat, que_mask, opt_feat, opt_mask, opt_sum, closest_sent_feat, csf_mask, cfg):
        que_f = que_feat
        opt_f = opt_feat
        csf = closest_sent_feat

        att_q2c = torch.matmul(que_f,
                               csf.reshape(-1, cfg.max_sent_para * cfg.max_words_sent, cfg.word_emb).permute(0, 2, 1))
        # att_q2c = att_q2c / math.sqrt(que_f.shape[-1])
        # att_q2c = att_q2c.masked_fill(csf_mask.reshape(-1, cfg.max_sent_para * cfg.max_words_sent).unsqueeze(1) == 1,
        #                               -9e15)
        # att_q2c = F.softmax(att_q2c, dim=-1)
        # att_q2c = att_q2c.masked_fill(que_mask.unsqueeze(-1) == 1, 0.)
        att_q2c = torch.sum(att_q2c, dim=1).reshape(-1, cfg.max_sent_para, cfg.max_words_sent)
        att_q2c = torch.sum(att_q2c, dim=-1)
        csf_len = torch.sum(~(csf_mask), dim=-1)
        csf_len = csf_len.masked_fill(csf_len == 0, -9e15).float()
        att_q2c_sent = torch.div(att_q2c, csf_len)
        _, ix_q2c = torch.max(att_q2c_sent, dim=-1)
        que_csf = torch.cat([csf[i][int(id)] for i, id in enumerate(ix_q2c)], dim=0)
        que_csf = que_csf.reshape(-1, cfg.max_words_sent, cfg.word_emb)

        que_csf = que_csf.repeat(1, cfg.max_opt_count, 1).reshape(-1, cfg.max_opt_count, cfg.max_words_sent,
                                                                  cfg.word_emb)
        que_csf = que_csf.reshape(-1, cfg.max_opt_count, cfg.max_words_sent * cfg.word_emb)
        que_csf = que_csf.masked_fill(opt_sum.unsqueeze(-1) == 1, 0.)
        que_csf = que_csf.reshape(-1, cfg.max_opt_count, cfg.max_words_sent, cfg.word_emb)

        csf_ori = csf
        csf = csf.repeat(1, cfg.max_opt_count, 1, 1).reshape(-1, cfg.max_opt_count, cfg.max_sent_para,
                                                             cfg.max_words_sent, cfg.word_emb).reshape(-1,
                                                                                                       cfg.max_opt_count,
                                                                                                       cfg.max_sent_para * cfg.max_words_sent,
                                                                                                       cfg.word_emb)
        att_o2c = torch.matmul(opt_f, csf.permute(0, 1, 3, 2))
        # att_o2c = att_o2c / math.sqrt(opt_feat.shape[-1])
        # att_o2c = att_o2c.masked_fill(
        #     csf_mask.reshape(-1, cfg.max_sent_para * cfg.max_words_sent).unsqueeze(1).unsqueeze(2) == 1, -9e15)
        # att_o2c = F.softmax(att_o2c, dim=-1)
        # att_o2c = att_o2c.masked_fill(opt_mask.unsqueeze(-1) == 1, 0.)
        att_o2c = torch.sum(att_o2c, dim=2)
        att_o2c = att_o2c.reshape(-1, cfg.max_opt_count, cfg.max_sent_para, cfg.max_words_sent)
        att_o2c = torch.sum(att_o2c, dim=-1)
        csf_len = csf_len.repeat(1, cfg.max_opt_count).reshape(-1, cfg.max_opt_count, cfg.max_sent_para)
        att_o2c_sent = torch.div(att_o2c, csf_len)
        _, ix_o2c = torch.max(att_o2c_sent, dim=-1)
        opt_csf = self.get_opt2csf(csf_ori, ix_o2c, cfg, opt_sum)

        que_csf_mask = make_mask(que_csf)
        opt_csf_mask = make_mask(opt_csf)
        return que_csf, que_csf_mask, opt_csf, opt_csf_mask

    def get_opt2csf(self, csf, ix, cfg, opt_sum_csf):
        batch_size = csf.shape[0]
        batch_opt_csf_list = []

        for i in range(batch_size):
            opt_csf_list = []
            for j in range(cfg.max_opt_count):
                opt_csf_list.append(csf[0][ix[i][j]])

            csf_opt = torch.cat(opt_csf_list, dim=0)
            csf_opt = csf_opt.reshape(cfg.max_opt_count, cfg.max_words_sent, -1)
            batch_opt_csf_list.append(csf_opt)
        batch_opt_csf = torch.cat(batch_opt_csf_list, dim=0)
        batch_opt_csf = batch_opt_csf.reshape(batch_size, cfg.max_opt_count, cfg.max_words_sent, -1)
        batch_opt_csf = batch_opt_csf.reshape(batch_size, cfg.max_opt_count, -1)
        batch_opt_csf = batch_opt_csf.masked_fill(opt_sum_csf.unsqueeze(-1) == 1, 0.)
        batch_opt_csf = batch_opt_csf.reshape(batch_size, cfg.max_opt_count, cfg.max_words_sent, -1)
        return batch_opt_csf

class DiagramNet(nn.Module):
    def __init__(self, cfg):
        super(DiagramNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.init_word_emb,
            hidden_size=cfg.lstm_hid,
            num_layers=cfg.lstm_layer,
            batch_first=True,
        )

        self.att = MultiSA(cfg)

        self.gat = GAT(
            cfg.max_diagram_node,
            cfg.word_emb,
            cfg.word_emb,
            cfg.gat_dropout,
            cfg.gat_alpha,
            cfg.gat_heads
        )

        self.intra2inter = INTRA_2_INTER(cfg)

        self.flat = AttFlat(cfg)

        self.ln = LayerNorm(cfg.mlp_out)
        # self.classify = nn.CosineSimilarity(dim=-1)
        self.classify = nn.Linear(cfg.mlp_out * 4, 1)

    def forward(self, que_emb, opt_emb, dq_matrix, dq_node_emb, dd_matrix, dd_node_emb, closest_sent_emb, cfg):
        batch_size = que_emb.shape[0]
        que_mask = make_mask(que_emb)
        que_feat, _ = self.lstm(que_emb)
        que_feat = que_feat.masked_fill(que_mask.unsqueeze(-1) == 1, 0.)
        que_feat = self.att(que_feat, que_mask.unsqueeze(1).unsqueeze(2))
        que_feat = que_feat.masked_fill(que_mask.unsqueeze(-1) == 1, 0.)

        opt_mask = make_mask(opt_emb)
        opt_sum = make_mask_num(opt_emb)
        opt_feat = opt_emb.reshape(-1, cfg.max_opt_len, cfg.init_word_emb)
        opt_feat, _ = self.lstm(opt_feat)
        opt_feat = opt_feat.masked_fill(opt_mask.reshape(-1, cfg.max_opt_len).unsqueeze(-1) == 1, 0.)
        opt_feat = self.att(opt_feat, opt_mask.reshape(batch_size * cfg.max_opt_count, -1).unsqueeze(1).unsqueeze(2))
        opt_feat = opt_feat.reshape(batch_size, cfg.max_opt_count, cfg.max_opt_len, -1)
        opt_feat = opt_feat.masked_fill(opt_mask.unsqueeze(-1) == 1, 0.)

        closest_sent_mask = make_mask(closest_sent_emb)
        closest_sent_sum = make_mask_num(closest_sent_emb)
        closest_sent_feat = closest_sent_emb.reshape(-1, cfg.max_words_sent, cfg.init_word_emb)
        closest_sent_feat, _ = self.lstm(closest_sent_feat)
        closest_sent_feat = closest_sent_feat.masked_fill(
            closest_sent_mask.reshape(-1, cfg.max_words_sent).unsqueeze(-1) == 1, 0.)
        closest_sent_feat = self.att(closest_sent_feat,
                                     closest_sent_mask.reshape(-1, cfg.max_words_sent).unsqueeze(1).unsqueeze(2))
        closest_sent_feat = closest_sent_feat.reshape(batch_size, cfg.max_sent_para, cfg.max_words_sent, -1)
        closest_sent_feat = closest_sent_feat.masked_fill(closest_sent_mask.unsqueeze(-1) == 1, 0.)

        dd_node_mask = make_mask(dd_node_emb)
        dd_node_feat = self.gat(dd_node_emb, dd_matrix, cfg)
        dq_node_mask = make_mask(dq_node_emb)
        dq_node_feat = self.gat(dq_node_emb, dq_matrix, cfg)

        que_feat, dq_node_feat = self.intra2inter(que_feat,
                                                  que_mask.unsqueeze(1).unsqueeze(2),
                                                  dq_node_feat,
                                                  dq_node_mask.unsqueeze(1).unsqueeze(2))

        que_csf, que_csf_mask, opt_csf, opt_csf_mask = self.find_context(que_feat,
                                                                         que_mask,
                                                                         opt_feat,
                                                                         opt_mask,
                                                                         opt_sum,
                                                                         closest_sent_feat,
                                                                         closest_sent_mask,
                                                                         cfg)

    def find_context(self, que_feat, que_mask, opt_feat, opt_mask, opt_sum, closest_sent_feat, csf_mask, cfg):
        que_f = que_feat
        opt_f = opt_feat
        csf = closest_sent_feat

        att_q2c = torch.matmul(que_f,
                               csf.reshape(-1, cfg.max_sent_para * cfg.max_words_sent, cfg.word_emb).permute(0, 2, 1))
        # att_q2c = att_q2c / math.sqrt(que_f.shape[-1])
        # att_q2c = att_q2c.masked_fill(csf_mask.reshape(-1, cfg.max_sent_para * cfg.max_words_sent).unsqueeze(1) == 1,
        #                               -9e15)
        # att_q2c = F.softmax(att_q2c, dim=-1)
        # att_q2c = att_q2c.masked_fill(que_mask.unsqueeze(-1) == 1, 0.)
        att_q2c = torch.sum(att_q2c, dim=1).reshape(-1, cfg.max_sent_para, cfg.max_words_sent)
        att_q2c = torch.sum(att_q2c, dim=-1)
        csf_len = torch.sum(~(csf_mask), dim=-1)
        csf_len = csf_len.masked_fill(csf_len == 0, -9e15).float()
        att_q2c_sent = torch.div(att_q2c, csf_len)
        _, ix_q2c = torch.max(att_q2c_sent, dim=-1)
        que_csf = torch.cat([csf[i][int(id)] for i, id in enumerate(ix_q2c)], dim=0)
        que_csf = que_csf.reshape(-1, cfg.max_words_sent, cfg.word_emb)

        que_csf = que_csf.repeat(1, cfg.max_opt_count, 1).reshape(-1, cfg.max_opt_count, cfg.max_words_sent,
                                                                  cfg.word_emb)
        que_csf = que_csf.reshape(-1, cfg.max_opt_count, cfg.max_words_sent * cfg.word_emb)
        que_csf = que_csf.masked_fill(opt_sum.unsqueeze(-1) == 1, 0.)
        que_csf = que_csf.reshape(-1, cfg.max_opt_count, cfg.max_words_sent, cfg.word_emb)

        csf_ori = csf
        csf = csf.repeat(1, cfg.max_opt_count, 1, 1).reshape(-1, cfg.max_opt_count, cfg.max_sent_para,
                                                             cfg.max_words_sent, cfg.word_emb).reshape(-1,
                                                                                                       cfg.max_opt_count,
                                                                                                       cfg.max_sent_para * cfg.max_words_sent,
                                                                                                       cfg.word_emb)
        att_o2c = torch.matmul(opt_f, csf.permute(0, 1, 3, 2))
        # att_o2c = att_o2c / math.sqrt(opt_feat.shape[-1])
        # att_o2c = att_o2c.masked_fill(
        #     csf_mask.reshape(-1, cfg.max_sent_para * cfg.max_words_sent).unsqueeze(1).unsqueeze(2) == 1, -9e15)
        # att_o2c = F.softmax(att_o2c, dim=-1)
        # att_o2c = att_o2c.masked_fill(opt_mask.unsqueeze(-1) == 1, 0.)
        att_o2c = torch.sum(att_o2c, dim=2)
        att_o2c = att_o2c.reshape(-1, cfg.max_opt_count, cfg.max_sent_para, cfg.max_words_sent)
        att_o2c = torch.sum(att_o2c, dim=-1)
        csf_len = csf_len.repeat(1, cfg.max_opt_count).reshape(-1, cfg.max_opt_count, cfg.max_sent_para)
        att_o2c_sent = torch.div(att_o2c, csf_len)
        _, ix_o2c = torch.max(att_o2c_sent, dim=-1)
        opt_csf = self.get_opt2csf(csf_ori, ix_o2c, cfg, opt_sum)

        que_csf_mask = make_mask(que_csf)
        opt_csf_mask = make_mask(opt_csf)
        return que_csf, que_csf_mask, opt_csf, opt_csf_mask

    def get_opt2csf(self, csf, ix, cfg, opt_sum_csf):
        batch_size = csf.shape[0]
        batch_opt_csf_list = []

        for i in range(batch_size):
            opt_csf_list = []
            for j in range(cfg.max_opt_count):
                opt_csf_list.append(csf[0][ix[i][j]])

            csf_opt = torch.cat(opt_csf_list, dim=0)
            csf_opt = csf_opt.reshape(cfg.max_opt_count, cfg.max_words_sent, -1)
            batch_opt_csf_list.append(csf_opt)
        batch_opt_csf = torch.cat(batch_opt_csf_list, dim=0)
        batch_opt_csf = batch_opt_csf.reshape(batch_size, cfg.max_opt_count, cfg.max_words_sent, -1)
        batch_opt_csf = batch_opt_csf.reshape(batch_size, cfg.max_opt_count, -1)
        batch_opt_csf = batch_opt_csf.masked_fill(opt_sum_csf.unsqueeze(-1) == 1, 0.)
        batch_opt_csf = batch_opt_csf.reshape(batch_size, cfg.max_opt_count, cfg.max_words_sent, -1)
        return batch_opt_csf
