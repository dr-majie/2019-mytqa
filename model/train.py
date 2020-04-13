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
from model.config import ConfigBeta
from data.textual_data_loader import TextualDataset
from model.net import TextualNetBeta
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
from model.test import test_engine
from utils.util import print_obj


def run_textual_net(cfg):
    net = TextualNetBeta(cfg)
    net.cuda()
    net.train()
    criterion = CrossEntropyLoss(reduction='sum')
    optimizer = Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_dataset = TextualDataset(cfg)
    train_dataloader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    cfg.mode = 'test'
    val_dataset = TextualDataset(cfg)

    for epoch in range(cfg.max_epochs):
        loss_sum = 0
        ques_sum = 0
        correct_sum = 0
        for step, (
                que_iter,
                opt_iter,
                ans_iter,
                cs_iter
        ) in enumerate(train_dataloader):
            que_iter = que_iter.cuda()
            opt_iter = opt_iter.cuda()
            ans_iter = ans_iter.cuda()
            cs_iter = cs_iter.cuda()

            optimizer.zero_grad()
            # que_emb, opt_emb, adjacency_matrices, node_emb, cfg
            pred = net(
                que_iter,
                opt_iter,
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
            a = [x.grad for x in optimizer.param_groups[0]['params']]
            optimizer.step()

            correct_sum += label_ix.eq(pred_ix).cpu().sum()
            ques_sum += que_iter.shape[0]
        correct_sum = np.array(correct_sum, dtype='float32')
        accuracy = correct_sum / float(ques_sum)
        print('epoch:', epoch, 'training loss {}'.format(loss_sum), 'correct sum:', correct_sum, 'total questions:',
              ques_sum, 'accuracy: {}'.format(accuracy))
        # print(net.state_dict()['classify.weight'])
        test_engine(net.state_dict(), cfg, val_dataset)
        scheduler.step()



def run_diagram_net(cfg):
    pass


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

    cfg = ConfigBeta()
    args_dict = cfg.parse_to_dict(args)
    cfg.add_attr(args_dict)
    print_obj(cfg)

    if cfg.model == 'tn':
        run_textual_net(cfg)
    else:
        run_diagram_net(cfg)
