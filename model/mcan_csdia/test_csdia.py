# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/4/1 14:38
# @Author:Ma Jie
# @FileName: test.py
# -----------------------------------------------
import torch, argparse, random
import numpy as np
import torch.utils.data as Data
from torch.nn import CrossEntropyLoss
from model.mcan_csdia.net import Net
from data.diagram_data_loader import DiagramDataset
from utils.util import print_obj
from model.mcan_csdia.config import Cfgs


def test_engine(state_dict=None, cfg=None, dataset=None):
    net = Net(cfg)
    net.eval()
    net.cuda()
    flag = 'val'
    if state_dict == None:
        flag = 'test'
        dataset = DiagramDataset(cfg)
        net.load_state_dict(torch.load(cfg.save_path + '/' + cfg.csdia_t + '/mcan/' + cfg.version +
                                       '/epoch' + cfg.epoch + '.pkl'))
    else:
        net.load_state_dict(state_dict)

    criterion = CrossEntropyLoss()
    print('Note: begin to test the model using ' + flag + ' split')
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
            dia_f_iter,
            opt_iter,
            dia_mat_iter,
            dia_nod_emb_iter,
            ans_iter
    ) in enumerate(dataloader):
        que_iter = que_iter.cuda()
        dia_f_iter = dia_f_iter.cuda()
        opt_iter = opt_iter.cuda()
        dia_mat_iter = dia_mat_iter.cuda()
        dia_nod_emb_iter = dia_nod_emb_iter.cuda()
        ans_iter = ans_iter.cuda()

        with torch.no_grad():
            pred = net(
                que_iter,
                dia_f_iter,
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


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--csdia_t', dest='csdia_t', choices=['mc', 'tf'], type=str, required=True)
    parse.add_argument('--dataset', dest='dataset', choices=['tqa', 'csdia'], type=str, required=True)
    parse.add_argument('--version', dest='version', type=str, required=True)
    parse.add_argument('--epoch', dest='epoch', type=str, required=True)
    parse.add_argument('--splits', dest='splits', default='test', type=str)
    parse.add_argument('--no-cuda', action='store_true', default=False, help='Disable cuda training')
    parse.add_argument('--seed', type=int, default=72, help='Random seed.')

    args = parse.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cfg = Cfgs(args.csdia_t)
    args_dict = cfg.parse_to_dict(args)
    cfg.add_attr(args_dict)
    print_obj(cfg)

    test_engine(cfg=cfg)
