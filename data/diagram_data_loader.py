# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/27 10:23
# @Author:Ma Jie
# @FileName: diagram_data_loader.py
# -----------------------------------------------

import torch.utils.data as Data
from utils.util import load_diagram_data, load_csdia_data


class DiagramDataset(Data.Dataset):
    def __init__(self, cfg):
        if cfg.dataset == 'tqa':
            self.que, self.opt, self.dq_matrix, self.dq_node_emb, self.dd_matrix, self.dd_node_emb, \
            self.ans, self.closest_sent = load_diagram_data(cfg)
        else:
            self.que, self.opt, self.dq_matrix, self.dq_node_emb, self.ans = load_csdia_data(cfg)

        self.data_size = self.que.__len__()
        print('data_size: {}'.format(self.data_size))

    def __getitem__(self, idx):
        que_iter = self.que[idx]
        opt_iter = self.opt[idx]
        dq_matrix_iter = self.dq_matrix[idx]
        dq_node_emb_iter = self.dq_node_emb[idx]
        dd_matrix_iter = self.dd_matrix[idx]
        dd_node_emb_iter = self.dd_node_emb[idx]
        ans_iter = self.ans[idx]
        cs_iter = self.closest_sent[idx]

        return que_iter, opt_iter, dq_matrix_iter, dq_node_emb_iter, dd_matrix_iter, \
               dd_node_emb_iter, ans_iter, cs_iter

    def __len__(self):
        return len(self.que)
