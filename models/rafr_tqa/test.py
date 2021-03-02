# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/4/1 14:38
# @Author:Ma Jie
# @FileName: test.py
# -----------------------------------------------
import torch
import numpy as np
from data.textual_data_loader import TextualDataset
import torch.utils.data as Data
from torch.nn import CrossEntropyLoss


def test_engine(net, cfg):
    net.eval()
    cfg.mode = 'test'
    if cfg.model == 'tn':
        dataset = TextualDataset(cfg)
        print('Note: begin to test the model')
        dataloader = Data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        criterion = CrossEntropyLoss(reduction='sum')
        ques_sum = 0
        correct_num = 0
        loss_sum = 0
        for step, (
                que_iter,
                opt_iter,
                ans_iter,
                adj_matrices_iter,
                node_emb_iter
        ) in enumerate(dataloader):
            que_iter = que_iter.cuda()
            opt_iter = opt_iter.cuda()
            ans_iter = ans_iter.cuda()
            adj_matrices_iter = adj_matrices_iter.cuda()
            node_emb_iter = node_emb_iter.cuda()

            pred = net(
                que_iter,
                opt_iter,
                adj_matrices_iter,
                node_emb_iter,
                cfg
            )

            ques_sum += que_iter.shape[0]
            _, pred_idx = torch.max(pred, -1)
            _, label = torch.max(ans_iter, -1)

            val_label = label
            val_label = val_label.squeeze(-1)
            loss = criterion(pred, val_label)
            loss_sum += loss

            batch_size = label.shape[0]
            label = torch.reshape(label, (-1, batch_size)).squeeze(0)
            correct_num += label.eq(pred_idx).sum()

        correct_num = np.array(correct_num.cpu(), dtype=float)
        accuracy = correct_num / ques_sum
        print('val loss {}'.format(loss_sum), '* correct prediction:', correct_num, '  * total questions:', ques_sum,
              '  * accuracy is {}'.format(accuracy), '\n')
    else:
        pass
