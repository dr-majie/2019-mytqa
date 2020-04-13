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
from model.net import TextualNetBeta


def test_engine(state_dict, cfg, dataset):
    net = TextualNetBeta(cfg)
    net.eval()
    net.cuda()
    if state_dict is None:
        print('state dict is none')
    else:
        net.load_state_dict(state_dict)
    # print(net.state_dict()['classify.weight'])
    criterion = CrossEntropyLoss(reduction='sum')
    if cfg.model == 'tn':
        print('Note: begin to test the model')
        dataloader = Data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        ques_sum = 0
        correct_sum = 0
        loss_sum = 0
        for step, (
                que_iter,
                opt_iter,
                ans_iter,
                cs_iter
        ) in enumerate(dataloader):
            que_iter = que_iter.cuda()
            opt_iter = opt_iter.cuda()
            ans_iter = ans_iter.cuda()
            cs_iter = cs_iter.cuda()

            with torch.no_grad():
                pred = net(
                    que_iter,
                    opt_iter,
                    cs_iter,
                    cfg
                )

                ques_sum += que_iter.shape[0]
                _, pred_idx = torch.max(pred, -1)
                _, label_ix = torch.max(ans_iter, -1)

                label_ix = label_ix.squeeze(-1)
                loss = criterion(pred, label_ix)
                loss_sum += loss
                correct_sum += label_ix.eq(pred_idx).cpu().sum()

        correct_sum = np.array(correct_sum, dtype='float32')
        accuracy = correct_sum / float(ques_sum)
        # print(net.state_dict()['classify.weight'])

        print('val loss {}'.format(loss_sum), '* correct prediction:', correct_sum, '  * total questions:', ques_sum,
              '  * accuracy is {}'.format(accuracy), '\n')
    else:
        pass
