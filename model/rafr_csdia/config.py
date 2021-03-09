# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/26 8:21
# @Author:Ma Jie
# @FileName: config.py
# -----------------------------------------------
from types import MethodType


class ConfigBeta(object):
    def __init__(self, flag):
        self.gat_dropout = 0.0
        self.gat_hidden = 64
        self.gat_alpha = 0.2
        self.gat_heads = 8
        self.init_word_emb = 768

        self.mlp_in = 512
        self.mlp_hid = 256
        self.glimpse = 1
        self.mlp_dropout = 0.2
        self.mlp_out = 128

        self.lr = 0.0005
        self.weight_decay = 0.01  # 5e-4
        self.batch_size = 8
        self.num_workers = 8
        self.max_epochs = 60

        if flag in 'mc':
            self.opt_num = 4
        else:
            self.opt_num = 2
        self.pre_path = '/data/majie/majie/codehub/2019-mytqa/processed_csdia_data/'
        self.save_path = '/data/kf/majie/codehub/2019-mytqa/saved'
        self.load_model = False

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
