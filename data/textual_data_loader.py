# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/27 10:23
# @Author:Ma Jie
# @FileName: textual_data_loader.py
# -----------------------------------------------
import torch.utils.data as Data
from utils.util import load_texutal_data


class TextualDataset(Data.Dataset):
    def __init__(self, cfg):
        if cfg.mode == 'train':
            self.que, self.opt, self.ans, self.adj_matrices, self.node_emb, _ = load_texutal_data(cfg)
        else:
            self.que, self.opt, self.ans, self.adj_matrices, self.node_emb, _ = load_texutal_data(cfg)

    def __getitem__(self, idx):
        que_iter = self.que[idx]
        opt_iter = self.opt[idx]
        ans_iter = self.ans[idx]
        adj_matrices_iter = self.adj_matrices[idx]
        node_emb_iter = self.node_emb[idx]

        return que_iter, opt_iter, ans_iter, adj_matrices_iter, node_emb_iter

    def __len__(self):
        assert len(self.que) == len(self.opt) == len(self.ans) == len(self.adj_matrices) == len(
            self.node_emb), 'data size of each iter is not equal.'
        return len(self.que)
