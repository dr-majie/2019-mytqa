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
from model.net import TextualNetBeta, DiagramNet
from utils.util import count_accurate_prediction_text


def test_engine(state_dict, cfg, dataset):
    if cfg.model == 'tn':
        net = TextualNetBeta(cfg)
        net.eval()
        net.cuda()
        if state_dict is None:
            print('state dict is none')
        else:
            net.load_state_dict(state_dict)
        # print(net.state_dict()['classify.weight'])
        criterion = CrossEntropyLoss(reduction='sum')
        print('Note: begin to test the model')
        dataloader = Data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        ques_sum_tf = 0
        ques_sum_mc = 0

        correct_sum_tf = 0
        correct_sum_mc = 0
        loss_sum = 0
        for step, (
                que_iter,
                opt_iter,
                ans_iter,
                cs_iter,
                qt_iter
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

                _, pred_ix = torch.max(pred, -1)
                _, label_ix = torch.max(ans_iter, -1)

                label_ix = label_ix.squeeze(-1)
                loss = criterion(pred, label_ix)
                loss_sum += loss

                a, b, c, d = count_accurate_prediction_text(label_ix, pred_ix, qt_iter)
                correct_sum_tf += a
                correct_sum_mc += b

                ques_sum_tf += c
                ques_sum_mc += d

        correct_sum = correct_sum_tf + correct_sum_mc
        total_que = ques_sum_mc + ques_sum_tf
        tf_acc = correct_sum_tf / ques_sum_tf
        mc_acc = correct_sum_mc / ques_sum_mc
        overall_acc = correct_sum / total_que

        print(40 * '*', '\n',
              'loss: {}'.format(loss_sum), '\n',
              'correct sum:', correct_sum, '\n',
              'total questions:', total_que, '\n',
              'tf_accuracy:', tf_acc, '\n',
              'mc_accuracy:', mc_acc, '\n',
              'overall accuracy:', overall_acc)
        print(40 * '*')
        print('\n')
    else:
        net = DiagramNet(cfg)
        net.eval()
        net.cuda()
        if state_dict is None:
            print('state dict is none')
        else:
            net.load_state_dict(state_dict)
        # print(net.state_dict()['classify.weight'])
        criterion = CrossEntropyLoss(reduction='sum')
        print('Note: begin to test the model')
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
                dq_matrix_iter,
                dq_node_emb_iter,
                dd_matrix_iter,
                dd_node_emb_iter,
                ans_iter,
                cs_iter
        ) in enumerate(dataloader):
            que_iter = que_iter.cuda()
            opt_iter = opt_iter.cuda()
            dq_matrix_iter = dq_matrix_iter.cuda()
            dq_node_emb_iter = dq_node_emb_iter.cuda()
            dd_matrix_iter = dd_matrix_iter.cuda()
            dd_node_emb_iter = dd_node_emb_iter.cuda()
            ans_iter = ans_iter.cuda()
            cs_iter = cs_iter.cuda()

            with torch.no_grad():
                pred = net(
                    que_iter,
                    opt_iter,
                    dq_matrix_iter,
                    dq_node_emb_iter,
                    dd_matrix_iter,
                    dd_node_emb_iter,
                    cs_iter,
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
              'loss: {}'.format(loss_sum), '\n',
              'correct sum:', correct_sum, '\n',
              'total questions:', que_sum, '\n',
              'overall accuracy:', overall_acc)
        print(40 * '*')
        print('\n')
