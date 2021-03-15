# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/4/1 14:38
# @Author:Ma Jie
# @FileName: test.py
# -----------------------------------------------
import torch
import numpy as np
import torch.utils.data as Data
from torch.nn import CrossEntropyLoss
from model.rafr_csdia.net import Net


def test_engine(state_dict, cfg, dataset):
    net = Net(cfg)
    net.eval()
    net.cuda()
    flag = 'val'
    if cfg.load_model == True:
        flag = 'test'
        net.load_state_dict(torch.load(cfg.save_path + '/epoch' + str(0) + '.pkl'))
    else:
        net.load_state_dict(state_dict)

    criterion = CrossEntropyLoss()
    print('Note: begin to test the model using' + flag + 'dataset')
    dataloader = Data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    loss_sum = 0
    correct_sum = 0
    que_sum = 0
    for step, (
            que_iter,
            opt_iter,
            dia_mat_iter,
            dia_nod_emb_iter,
            ans_iter
    ) in enumerate(dataloader):
        que_iter = que_iter.cuda()
        opt_iter = opt_iter.cuda()
        dia_mat_iter = dia_mat_iter.cuda()
        dia_nod_emb_iter = dia_nod_emb_iter.cuda()
        ans_iter = ans_iter.cuda()

        with torch.no_grad():
            pred = net(
                que_iter,
                opt_iter,
                dia_mat_iter,
                dia_nod_emb_iter,
                cfg
            )

            _, pred_ix = torch.max(pred, -1)
            _, label_ix = torch.max(ans_iter, -1)

            label_ix = label_ix.squeeze(-1)
            loss = criterion(pred, label_ix)
            # print(loss)
            loss_sum += loss

            correct_sum += label_ix.eq(pred_ix).cpu().sum()
            que_sum += que_iter.shape[0]
    correct_sum = np.array(correct_sum, dtype='float32')
    overall_acc = correct_sum / que_sum

    print(40 * '*', '\n',
          'loss: {}'.format(loss_sum / que_sum), '\n',
          'correct sum:', correct_sum, '\n',
          'total questions:', que_sum, '\n',
          'overall accuracy:', overall_acc)
    print(40 * '*')
    print('\n')
