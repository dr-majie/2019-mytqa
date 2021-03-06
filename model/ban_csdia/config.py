# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/8/15 17:39
# @Author:Ma Jie
# -----------------------------------------------
import os, random
from types import MethodType


class Cfgs():
    def __init__(self, flag):
        super(Cfgs, self).__init__()
        self.classifer_dropout_r = 0.5
        self.dropout_r = 0.2
        self.dia_feat_size = 2048  # the feature of the diagram
        self.flat_out_size = 768  # flatten hidden size
        self.flat_mlp_size = 512  # flatten one tensor to 512-dimensional tensor
        self.flat_glimpse = 1
        self.hidden_size = 768
        self.glimpse = 1
        self.gpu = '0'
        self.k_times = 3
        self.ba_hidden_size = self.k_times * self.hidden_size
        self.not_fine_tuned = 'False'
        self.simclr = {'base_model': 'resnet50', 'out_dim': 128,
                       'checkpoints_folder': '/data/majie/majie/codehub/simclr/runs/Sep08_21-33-04_lthpc/checkpoints/'}
        self.lr = 0.0001
        self.weight_decay = 5e-4
        self.batch_size = 1
        self.num_workers = 8
        self.max_epochs = 10
        self.pre_path = '/data/majie/majie/codehub/2019-mytqa/processed_csdia_data/'
        self.save_path = '/data/majie/majie/codehub/2019-mytqa/saved/csdia'
        if flag in 'mc':
            self.opt_num = 4
        else:
            self.opt_num = 2

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.load_model = False
        self.version = str(random.randint(0, 999))

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
