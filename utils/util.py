# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/24 20:05
# @Author:Ma Jie
# @FileName: util.py
# -----------------------------------------------
import os
import numpy as np
import pickle
import torch


def get_list_of_dirs(dir_path):
    dirlist = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    dirlist.sort()
    return dirlist


def get_ndq_list_of_dirs(dir_path):
    ndq_list = [name for name in os.listdir(dir_path) if
                os.path.isdir(os.path.join(dir_path, name)) and name.startswith('N')]
    ndq_list.sort()
    return ndq_list


def get_dq_list_of_dirs(dir_path):
    dq_list = [name for name in os.listdir(dir_path) if
               os.path.isdir(os.path.join(dir_path, name)) and name.startswith('DQ')]
    dq_list.sort()
    return dq_list


def load_texutal_data(cfg):
    que_list = []
    opt_list = []
    adj_matrices_list = []
    node_emb_list = []
    ans_list = []
    closest_sent_list = []
    if cfg.mode == 'train':
        slice_paths = ['train', 'val']
    else:
        slice_paths = ['test']

    for slice_path in slice_paths:
        path = cfg.pre_path + slice_path + cfg.suf_path
        lesson_list = get_list_of_dirs(path)

        for lesson in lesson_list:
            ndq_list_path = os.path.join(path, lesson)
            ndq_list = get_ndq_list_of_dirs(ndq_list_path)

            for ndq in ndq_list:
                ndq_path = os.path.join(ndq_list_path, ndq)
                # load question.pkl
                with open(os.path.join(ndq_path, 'Question.pkl'), 'rb') as f_que:
                    que_emb = pickle.load(f_que).reshape(-1, cfg.word_emb)
                    x_dim = que_emb.shape[0]
                    pad_dim = cfg.max_que_len - x_dim
                    que_emb = np.pad(que_emb, ((0, pad_dim), (0, 0)), 'constant', constant_values=(0, 0))
                    que_list.append(que_emb)

                # load option embedding
                option = 'a'
                opt_embs = []
                adj_matrices = []
                node_embs = []
                while os.path.exists(os.path.join(ndq_path, option + '.pkl')):
                    with open(os.path.join(ndq_path, option + '.pkl'), 'rb') as f_opt:
                        opt_emb = pickle.load(f_opt).reshape(-1, cfg.word_emb)
                        x_dim = opt_emb.shape[0]
                        pad_dim = cfg.max_opt_len - x_dim
                        opt_emb = np.pad(opt_emb, ((0, pad_dim), (0, 0)), 'constant', constant_values=(0, 0))
                        opt_embs.append(opt_emb)

                    # load adjacency matrix of each [quetion, option]
                    with open(os.path.join(ndq_path, 'adjacency_matrix_' + option + '.pkl'), 'rb') as f_adj_matrix:
                        adj_matrix = pickle.load(f_adj_matrix)
                        dim = adj_matrix.shape[0]
                        idendity_matrix = np.identity(dim)
                        adj_matrix += idendity_matrix
                        pad_dim = cfg.gat_max_nodes - dim
                        adj_matrix = np.pad(adj_matrix, ((0, pad_dim), (0, pad_dim)), 'constant',
                                            constant_values=(0, 0))
                        adj_matrices.append(adj_matrix)

                    # load node embeddings of each graph
                    with open(os.path.join(ndq_path, 'node_embedding_' + option + '.pkl'), 'rb') as f_node_emb:
                        node_emb = pickle.load(f_node_emb)
                        expand_node_emb = np.zeros((cfg.gat_max_nodes, cfg.gat_node_emb))
                        for i, emb in enumerate(node_emb.values()):
                            expand_node_emb[i, :] = emb
                        node_embs.append(expand_node_emb)

                    assert len(opt_embs) == len(adj_matrices) == len(
                        node_embs), 'the length of option, adjacency matrices with option, and node emb is not equal'

                    option = chr(ord(option) + 1)
                difference = cfg.max_opt_count - len(opt_embs)
                # pad the number of options to max_opt_num
                if difference > 0:
                    for i in range(difference):
                        opt_emb = np.zeros((cfg.max_opt_len, cfg.word_emb))
                        opt_embs.append(opt_emb)

                        adj_matrix = np.zeros((cfg.gat_max_nodes, cfg.gat_max_nodes))
                        adj_matrices.append(adj_matrix)

                        node_emb = np.zeros((cfg.gat_max_nodes, cfg.gat_node_emb))
                        node_embs.append(node_emb)

                assert len(opt_embs) == len(adj_matrices) == len(
                    node_embs) == cfg.max_opt_count, 'the number of elements in these lists is not equal to the max option count'

                opt_list.append(opt_embs)
                adj_matrices_list.append(adj_matrices)
                node_emb_list.append(node_embs)
                # load correct answer
                with open(os.path.join(ndq_path, 'correct_answer.pkl'), 'rb') as f_ans:
                    ans_one_hot = pickle.load(f_ans).reshape(-1, cfg.max_opt_count)
                    ans_list.append(ans_one_hot)

                # load closest sentence embedding
                with open(os.path.join(ndq_path, 'closest_sent.pkl'), 'rb') as f_closest_sent:
                    closest_sent_emb = pickle.load(f_closest_sent)
                    closest_sent_list.append(closest_sent_emb)

    assert len(que_list) == len(opt_list) == len(ans_list) == len(adj_matrices_list) == len(node_emb_list) == len(
        closest_sent_list), 'the number of these list is not equal.'

    return torch.from_numpy(np.array(que_list, dtype='float32')), \
           torch.from_numpy(np.array(opt_list, dtype='float32')), \
           torch.from_numpy(np.array(ans_list, dtype='float32')), \
           torch.from_numpy(np.array(adj_matrices_list, dtype='float32')), \
           torch.from_numpy(np.array(node_emb_list, dtype='float32')), \
           torch.from_numpy(np.array(closest_sent_list, dtype='float32'))


def make_mask(feature):
    try:
        if feature.dim() == 3:
            return torch.sum(torch.abs(feature), dim=-1) == 0

        if feature.dim() == 4:
            return torch.sum((torch.sum(torch.abs(feature), dim=-1)), dim=-1) == 0
    except Exception as e:
        print('Note: this feature does not satisfy the required conditions.')
