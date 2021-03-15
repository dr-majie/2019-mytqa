# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/10/19 9:59
# @Author:Ma Jie
# -----------------------------------------------

import os, random
from types import MethodType


class Cfgs():
    def __init__(self, flag):
        super(Cfgs, self).__init__()
        self.hidden_size = 512
        self.dropout_r = 0.1
        self.dia_feat_size = 2048  # the feature of the diagram
        self.multi_head = 8  # Multi-head number in MCA layers
        # (Warning: HIDDEN_SIZE should be divided by MULTI_HEAD)
        self.layer = 6  # Model deeps
        # (Encoder and Decoder will be same deeps)
        self.flat_mlp_size = 512  # MLP size in flatten layers
        self.flat_glimpses = 1  # Flatten the last hidden to vector with {n} attention glimpses
        self.flat_out_size = 1024
        self.ff_size = int(self.hidden_size * 4)
        self.hidden_size_head = int(self.hidden_size / self.multi_head)

        self.gpu = '0'
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
