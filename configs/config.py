# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/3 18:27
# @Author:Ma Jie
# @FileName: config.py
# -----------------------------------------------
import os


class Config(object):
    def __init__(self):
        self.word_vec_size = 300
        self.max_que_length = 65
        self.max_opt_length = 25
        self.max_opt_count = 7
        self.max_sent_para = 10
        self.max_words_sent = 20
        self.nb_epoch = 50
        self.batch_size = 16
        self.steps_per_epoch_dq = 54
        self.validation_steps_dq = 16
        self.steps_per_epoch_ndq = 333
        self.validation_steps_ndq = 100
        self.seed = 72
        self.train_data_path = 'data/train/processed_data/one_hot_files'
        self.val_data_path = 'data/val/processed_data/one_hot_files'
        self.test_data_path = 'data/test/processed_data/one_hot_files'
        self.word2vec_path = 'word2vec/GoogleNews-vectors-negative300.bin.gz'
        self.models_path = os.path.join('../data/train', 'saved_models')
