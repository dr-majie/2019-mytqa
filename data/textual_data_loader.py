# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/27 10:23
# @Author:Ma Jie
# @FileName: textual_data_loader.py
# -----------------------------------------------
import torch.utils.data as Data
from utils.util import load_texutal_data_beta


class TextualDataset(Data.Dataset):
    def __init__(self, cfg):
        # self.que, self.opt, self.ans, self.adj_matrices, self.node_emb, _ = load_texutal_data(cfg)
        self.que, self.opt, self.ans, self.closest_sent, self.que_type = load_texutal_data_beta(cfg)
        self.data_size = self.que.__len__()
        print('data_size: {}'.format(self.data_size))

    def __getitem__(self, idx):
        que_iter = self.que[idx]
        opt_iter = self.opt[idx]
        ans_iter = self.ans[idx]
        # adj_matrices_iter = self.adj_matrices[idx]
        # node_emb_iter = self.node_emb[idx]
        cs_iter = self.closest_sent[idx]
        qt_iter = self.que_type[idx]
        # return que_iter, opt_iter, ans_iter, adj_matrices_iter, node_emb_iter
        return que_iter, opt_iter, ans_iter, cs_iter, qt_iter

    def __len__(self):
        assert len(self.que) == len(self.opt) == len(self.ans) == len(
            self.closest_sent), 'data size of each iter is not equal.'
        return len(self.que)
