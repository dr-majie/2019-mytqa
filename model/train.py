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
from model.config import Config
from data.textual_data_loader import TextualDataset
from model.net import TextualNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from model.test import test_engine


def run_textual_net(cfg):
    net = TextualNet(cfg)
    net.cuda()

    criterion = CrossEntropyLoss(reduction='sum')
    optimizer = Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    dataset = TextualDataset(cfg)
    dataloader = Data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    for epoch in range(cfg.max_epochs):
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

            optimizer.zero_grad()
            # que_emb, opt_emb, node_emb, adjacency_matrices, cfg
            pred = net(
                que_iter,
                opt_iter,
                adj_matrices_iter,
                node_emb_iter,
                cfg
            )

            # loss = criterion(pred, ans_iter)
            _, label = torch.max(ans_iter, -1)
            label = label.squeeze(-1)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
        print('epoch:', epoch, 'training loss {}'.format(loss))
        test_engine(net, cfg)
        cfg.mode = 'test'
        test_engine(net, cfg)
        cfg.mode = 'train'
        net.train()

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

    cfg = Config()
    args_dict = cfg.parse_to_dict(args)
    cfg.add_attr(args_dict)

    if cfg.model == 'tn':
        run_textual_net(cfg)
    else:
        run_diagram_net(cfg)
