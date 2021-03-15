# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2020/11/17 19:21
# @Author:wyx
# @File : layer.py
# -----------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
import os
from torch.nn.utils.weight_norm import weight_norm


class SimCLR(nn.Module):
    '''
    use simclr to obtain diagram representations.
    '''

    def __init__(self, cfgs):
        super(SimCLR, self).__init__()
        self.cfgs = cfgs

        spec = importlib.util.spec_from_file_location(
            "simclr",
            os.path.join(self.cfgs.simclr['checkpoints_folder'], 'resnet_simclr.py'))

        resnet_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resnet_module)
        self.simclr_resnet = resnet_module.ResNetSimCLR(self.cfgs.simclr['base_model'],
                                                        int(self.cfgs.simclr['out_dim']))
        # map_location must include 'cuda:0', otherwise it will occur error
        state_dict = torch.load(os.path.join(self.cfgs.simclr['checkpoints_folder'], 'model.pth'),
                                map_location='cuda:{}'.format(self.cfgs.gpu) if torch.cuda.is_available() else 'cpu')
        self.simclr_resnet.load_state_dict(state_dict)

        if self.cfgs.not_fine_tuned in 'True':
            for param in self.simclr_resnet.parameters():
                param.requires_grad = False

    def forward(self, dia):
        dia_feats, _ = self.simclr_resnet(dia)
        return dia_feats


class MFB(nn.Module):
    def __init__(self, cfgs, img_feat_size, ques_feat_size, is_first):
        super(MFB, self).__init__()
        self.cfgs = cfgs
        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, self.cfgs.mfb_k * self.cfgs.mfb_o)
        self.proj_q = nn.Linear(ques_feat_size, self.cfgs.mfb_k * self.cfgs.mfb_o)
        self.dropout = nn.Dropout(self.cfgs.dropout_r)
        self.pool = nn.AvgPool1d(self.cfgs.mfb_k, stride=self.cfgs.mfb_k)

    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, mfb_o)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)  # (N, C, K*O) (8,1,5120)
        ques_feat = self.proj_q(ques_feat)  # (N, 1, K*O) (8,1,5120)

        exp_out = img_feat * ques_feat  # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)  # (N, C, K*O)
        z = self.pool(exp_out) * self.cfgs.mfb_k  # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))  # (N, C*O)
        z = z.view(batch_size, -1, self.cfgs.mfb_o)  # (N, C, O) (8, 1, 1024)
        return z, exp_out


class QAtt(nn.Module):
    def __init__(self, cfgs):
        super(QAtt, self).__init__()
        self.cfgs = cfgs
        self.mlp = MLP(
            in_size=self.cfgs.lstm_out_size,
            mid_size=self.cfgs.hidden_size,
            out_size=self.cfgs.que_glimpse,
            dropout_r=self.cfgs.dropout_r,
            use_relu=True
        )

    def forward(self, ques_feat):
        '''
            ques_feat.size() -> (N, T, lstm_out_size)
            qatt_feat.size() -> (N, lstm_out_size * que_glimpse)
        '''
        qatt_maps = self.mlp(ques_feat)  # (N, T, que_glimpse) (8, 15, 2)
        qatt_maps = F.softmax(qatt_maps, dim=1)  # (N, T, que_glimpse)

        qatt_feat_list = []
        for i in range(self.cfgs.que_glimpse):
            mask = qatt_maps[:, :, i:i + 1]  # (N, T, 1)
            mask = mask * ques_feat  # (N, T, lstm_out_size)
            mask = torch.sum(mask, dim=1)  # (N, lstm_out_size)
            qatt_feat_list.append(mask)
        qatt_feat = torch.cat(qatt_feat_list, dim=1)  # (N, lstm_out_size*que_glimpse) (8, 2048)

        return qatt_feat


class IAtt(nn.Module):
    def __init__(self, cfgs, img_feat_size, ques_att_feat_size):
        super(IAtt, self).__init__()
        self.cfgs = cfgs
        self.dropout = nn.Dropout(self.cfgs.dropout_r)
        self.mfb = MFB(self.cfgs, img_feat_size, ques_att_feat_size, True)
        self.mlp = MLP(
            in_size=self.cfgs.mfb_o,
            mid_size=self.cfgs.hidden_size,
            out_size=self.cfgs.img_glimpse,
            dropout_r=self.cfgs.dropout_r,
            use_relu=True
        )

    def forward(self, img_feat, ques_att_feat):
        '''
            img_feats.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_att_feat.size() -> (N, lstm_out_size * que_glimpse)
            iatt_feat.size() -> (N, mfb_o * img_glimpse)
        '''
        ques_att_feat = ques_att_feat.unsqueeze(1)  # (N, 1, lstm_out_size * que_glimpse) (8,1,2048)
        img_feat = self.dropout(img_feat)  # (8,1,2048)
        z, _ = self.mfb(img_feat, ques_att_feat)  # (N, C, O)

        iatt_maps = self.mlp(z)  # (N, C, img_glimpse) (8,1,2)
        iatt_maps = F.softmax(iatt_maps, dim=1)  # (N, C, img_glimpse)

        iatt_feat_list = []
        for i in range(self.cfgs.img_glimpse):
            mask = iatt_maps[:, :, i:i + 1]  # (N, C, 1)
            mask = mask * img_feat  # (N, C, FRCN_FEAT_SIZE)
            mask = torch.sum(mask, dim=1)  # (N, FRCN_FEAT_SIZE)
            iatt_feat_list.append(mask)
        iatt_feat = torch.cat(iatt_feat_list, dim=1)  # (N, FRCN_FEAT_SIZE*img_glimpse) (8, 4096)

        return iatt_feat


class CoAtt(nn.Module):
    def __init__(self, cfgs):
        super(CoAtt, self).__init__()
        self.cfgs = cfgs

        img_feat_size = self.cfgs.dia_feat_size
        img_att_feat_size = img_feat_size * self.cfgs.img_glimpse
        ques_att_feat_size = self.cfgs.lstm_out_size * self.cfgs.que_glimpse

        self.q_att = QAtt(self.cfgs)
        self.i_att = IAtt(self.cfgs, img_feat_size, ques_att_feat_size)

        if self.cfgs.high_order:  # MFH
            self.mfh1 = MFB(self.cfgs, img_att_feat_size, ques_att_feat_size, True)
            self.mfh2 = MFB(self.cfgs, img_att_feat_size, ques_att_feat_size, False)
        else:  # mfb
            self.mfb = MFB(self.cfgs, img_att_feat_size, ques_att_feat_size, True)

    def forward(self, img_feat, ques_feat):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, lstm_out_size)
            z.size() -> MFH:(N, 2*O) / mfb:(N, O)
        '''
        ques_feat = self.q_att(ques_feat)  # (N, lstm_out_size*que_glimpse)
        fuse_feat = self.i_att(img_feat, ques_feat)  # (N, FRCN_FEAT_SIZE*img_glimpse)

        if self.cfgs.high_order:  # MFH
            z1, exp1 = self.mfh1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))  # z1:(N, 1, O)  exp1:(N, C, K*O)
            z2, _ = self.mfh2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)  # z2:(N, 1, O)  _:(N, C, K*O)
            z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)  # (N, 2*O)
        else:  # mfb
            z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))  # z:(N, 1, O)  _:(N, C, K*O) (8,1,1024)
            z = z.squeeze(1)  # (N, O) (8,1,1024)

        return z


# ------------------------------
# ----- Full Connect layer------
# ------------------------------
class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


# ------------------------------
# ----- Weight Normal MLP ------
# ------------------------------

class MLP_v1(nn.Module):
    """
    Simple class for non-linear fully connect network
    """

    def __init__(self, dims, act='ReLU', dropout_r=0.0):
        super(MLP_v1, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if dropout_r > 0:
                layers.append(nn.Dropout(dropout_r))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if act != '':
                layers.append(getattr(nn, act)())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# -----flatten tensor-----
# ------------------------

class FlattenAtt(nn.Module):
    def __init__(self, cfgs):
        super(FlattenAtt, self).__init__()
        self.cfgs = cfgs

        self.mlp = MLP_v1(
            dims=[cfgs.hidden_size, cfgs.flat_mlp_size, cfgs.flat_glimpse],
            act='',
            dropout_r=cfgs.dropout_r
        )

        self.linear_merge = weight_norm(
            nn.Linear(cfgs.hidden_size * cfgs.flat_glimpse, cfgs.flat_out_size),
            dim=None
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.cfgs.flat_glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted
