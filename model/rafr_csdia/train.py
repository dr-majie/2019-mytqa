# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/26 20:13
# @Author:Ma Jie
# @FileName: train.py
# -----------------------------------------------

import argparse
import torch
import random
import numpy as np
import torch.utils.data as Data
from model.rafr_tqa.config import ConfigBeta
from data.textual_data_loader import TextualDataset
from data.diagram_data_loader import DiagramDataset
from model.rafr_tqa.net import TextualNetBeta, DiagramNet
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler
from utils.util import print_obj, count_accurate_prediction_text

def run_diagram_net(cfg):
    net = DiagramNet(cfg)
    net.cuda()
    net.train()
    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_dataset = DiagramDataset(cfg)

    train_dataloader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    cfg.mode = 'test'
    val_dataset = DiagramDataset(cfg)

    for epoch in range(cfg.max_epochs):
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
        ) in enumerate(train_dataloader):
            que_iter = que_iter.cuda()
            opt_iter = opt_iter.cuda()
            dq_matrix_iter = dq_matrix_iter.cuda()
            dq_node_emb_iter = dq_node_emb_iter.cuda()
            dd_matrix_iter = dd_matrix_iter.cuda()
            dd_node_emb_iter = dd_node_emb_iter.cuda()
            ans_iter = ans_iter.cuda()
            cs_iter = cs_iter.cuda()

            optimizer.zero_grad()
            # que_emb, opt_emb, adjacency_matrices, node_emb, cfg
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
            # loss = criterion(pred, ans_iter)
            _, label_ix = torch.max(ans_iter, -1)
            _, pred_ix = torch.max(pred, -1)

            label_ix = label_ix.squeeze(-1)
            loss = criterion(pred, label_ix)
            loss_sum += loss

            loss.backward()
            clip_grad_norm_(net.parameters(), 10)
            optimizer.step()

            correct_sum += label_ix.eq(pred_ix).cpu().sum()
            que_sum += que_iter.shape[0]
        correct_sum = np.array(correct_sum, dtype='float32')
        overall_acc = correct_sum / que_sum

        print(40 * '=', '\n',
              'epoch:', epoch, '\n',
              'loss: {}'.format(loss_sum / que_sum), '\n',
              'correct sum:', correct_sum, '\n',
              'total questions:', que_sum, '\n',
              'accuracy:', overall_acc)
        print(40 * '=')
        print('\n')
        state = net.state_dict()
        torch.save(state,
                   cfg.save_path +
                   '/epoch' + str(epoch) +
                   'diagramNet' +
                   '.pkl'
                   )
        # test_engine(state, cfg, val_dataset)
        scheduler.step()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--mode', dest='mode', choices=['train', 'test'], type=str, required=True)
    parse.add_argument('--splits', dest='splits', choices=['train+val', 'train'], type=str, required=True)
    parse.add_argument('--no-cuda', action='store_true', default=False, help='Disable cuda training')
    parse.add_argument('--seed', type=int, default=72, help='Random seed.')
    parse.add_argument('--model', dest='model', choices=['tn', 'dn'],
                       help='tn denotes textual net, and dn denotes diagram net', type=str, required=True)
    args = parse.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cfg = ConfigBeta(args.model)
    args_dict = cfg.parse_to_dict(args)
    cfg.add_attr(args_dict)
    print_obj(cfg)

    if cfg.model == 'tn':
        run_textual_net(cfg)
    else:
        run_diagram_net(cfg)
