#-----------------------------------------------
#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time:2020/3/3 17:56
#@Author:Ma Jie
#@FileName: run.py
#-----------------------------------------------

from configs.config import Config
from preprocessing.data_prepare import prepare_data


def print_obj(obj):
    for item in obj.__dict__.items():
        print(item)

if __name__ == '__main__':

    print(28 * '*' + 'Hyperparameters' + '*' * 28)
    cfg = Config()
    print_obj(cfg)

    print('***************************Begin to process data***************************')
    train_data = prepare_data(cfg=cfg, processed_data_path=cfg.train_data_path, is_test_data=False)
    val_data = prepare_data(cfg=cfg, processed_data_path=cfg.val_data_path, is_test_data=False)
    test_data = prepare_data(cfg=cfg, processed_data_path=cfg.test_data_path, is_test_data=True)
