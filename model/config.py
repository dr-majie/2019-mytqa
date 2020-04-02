# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/26 8:21
# @Author:Ma Jie
# @FileName: config.py
# -----------------------------------------------
from types import MethodType


class Config(object):
    def __init__(self):
        self.gat_hid = 300
        self.gat_node_emb = 300
        self.gat_dropout = 0.0
        self.gat_alpha = 0.2
        self.gat_heads = 8
        self.gat_max_nodes = 60

        self.multi_hid = 300
        self.multi_heads = 6
        self.multi_dropout = 0.1

        self.word_emb = 300
        self.lstm_hid = 300
        self.lstm_layer = 1

        self.max_que_len = 30
        self.max_opt_len = 25
        self.max_opt_count = 7

        self.lr = 0.001
        self.weight_decay = 5e-4
        self.batch_size = 4
        self.num_workers = 8
        self.max_epochs = 300

        self.mlp_hid = 300
        self.glimpse = 1
        self.mlp_dropout = 0.1
        self.mlp_out = 600

        self.pre_path = '/data/kf/majie/wangyaxian/tqa/data/'
        self.suf_path = '/processed_data/one_hot_files/'

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)
        return args_dict

    def add_attr(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])
