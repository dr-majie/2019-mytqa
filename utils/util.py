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
import collections
import matplotlib.pyplot as plt

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


def get_dd_list_of_dirs(dir_path):
    dd_list = [name for name in os.listdir(dir_path) if
               os.path.isdir(os.path.join(dir_path, name)) and name.startswith('DD')]
    dd_list.sort()
    return dd_list


def load_texutal_data(cfg):
    que_list = []
    opt_list = []
    adj_matrices_list = []
    node_emb_list = []
    ans_list = []
    closest_sent_list = []
    if cfg.mode == 'train':
        if cfg.splits == 'train+val':
            slice_paths = ['train', 'val']
            print('Note: begin to load **', slice_paths, '**.')
        else:
            slice_paths = ['train']
            print('Note: begin to load **', slice_paths, '**.')
    else:
        if cfg.splits == 'train+val':
            slice_paths = ['test']
            print('Note: begin to load **', slice_paths, '**.')
        else:
            slice_paths = ['val']
            print('Note: begin to load **', slice_paths, '**.')

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
                    que_emb = pickle.load(f_que).reshape(-1, cfg.init_word_emb)
                    x_dim = que_emb.shape[0]
                    pad_dim = cfg.max_que_len - x_dim
                    if pad_dim >= 0:
                        que_emb = np.pad(que_emb, ((0, pad_dim), (0, 0)), 'constant', constant_values=(0, 0))
                    else:
                        que_emb = que_emb[:pad_dim]
                    que_list.append(que_emb)

                # load option embedding
                option = 'a'
                opt_embs = []
                adj_matrices = []
                node_embs = []
                while os.path.exists(os.path.join(ndq_path, option + '.pkl')):
                    with open(os.path.join(ndq_path, option + '.pkl'), 'rb') as f_opt:
                        opt_emb = pickle.load(f_opt).reshape(-1, cfg.init_word_emb)
                        x_dim = opt_emb.shape[0]
                        pad_dim = cfg.max_opt_len - x_dim
                        if pad_dim >= 0:
                            opt_emb = np.pad(opt_emb, ((0, pad_dim), (0, 0)), 'constant', constant_values=(0, 0))
                        else:
                            opt_emb = opt_emb[:pad_dim]
                        opt_embs.append(opt_emb)

                    # load adjacency matrix of each [quetion, option]
                    with open(os.path.join(ndq_path, 'adjacency_matrix_' + option + '.pkl'), 'rb') as f_adj_matrix:
                        adj_matrix = pickle.load(f_adj_matrix)
                        dim = adj_matrix.shape[0]
                        idendity_matrix = np.identity(dim)
                        adj_matrix += idendity_matrix
                        pad_dim = cfg.gat_max_nodes - dim
                        if pad_dim >= 0:
                            adj_matrix = np.pad(adj_matrix, ((0, pad_dim), (0, pad_dim)), 'constant',
                                                constant_values=(0, 0))
                        else:
                            adj_matrix = adj_matrix[:pad_dim, :pad_dim]
                        adj_matrices.append(adj_matrix)

                    # load node embeddings of each graph
                    with open(os.path.join(ndq_path, 'node_embedding_' + option + '.pkl'), 'rb') as f_node_emb:
                        node_emb = pickle.load(f_node_emb)
                        expand_node_emb = np.zeros((cfg.gat_max_nodes, cfg.gat_node_emb))
                        for i, emb in enumerate(node_emb.values()):
                            if i < cfg.gat_max_nodes:
                                expand_node_emb[i, :] = emb
                            else:
                                break
                        node_embs.append(expand_node_emb)

                    assert len(opt_embs) == len(adj_matrices) == len(
                        node_embs), 'the length of option, adjacency matrices with option, and node emb is not equal'

                    option = chr(ord(option) + 1)
                difference = cfg.max_opt_count - len(opt_embs)
                # pad the number of options to max_opt_num
                if difference > 0:
                    for i in range(difference):
                        opt_emb = np.zeros((cfg.max_opt_len, cfg.init_word_emb))
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
    return torch.sum(torch.abs(feature), dim=-1) == 0


def make_mask_num(feature):
    return torch.sum((torch.sum(torch.abs(feature), dim=-1)), dim=-1) == 0


def print_obj(obj):
    for item in obj.__dict__.items():
        print(item)


def load_texutal_data_beta(cfg):
    que_list = []
    que_type_list = []
    opt_list = []
    ans_list = []
    closest_sent_list = []
    if cfg.mode == 'train':
        if cfg.splits == 'train+val':
            slice_paths = ['train', 'val']
            print('Note: begin to load **', slice_paths, '**.')
        else:
            slice_paths = ['train']
            print('Note: begin to load **', slice_paths, '**.')
    else:
        if cfg.splits == 'train+val':
            slice_paths = ['test']
            print('Note: begin to load **', slice_paths, '**.')
        else:
            slice_paths = ['val']
            print('Note: begin to load **', slice_paths, '**.')

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
                    que_emb = pickle.load(f_que).reshape(-1, cfg.init_word_emb)
                    x_dim = que_emb.shape[0]
                    pad_dim = cfg.max_que_len - x_dim
                    if pad_dim >= 0:
                        que_emb = np.pad(que_emb, ((0, pad_dim), (0, 0)), 'constant', constant_values=(0, 0))
                    else:
                        que_emb = que_emb[:pad_dim]
                    que_list.append(que_emb)
                # load question type
                with open(os.path.join(ndq_path, 'question_type.pkl'), 'rb') as f_qt:
                    que_type = pickle.load(f_qt)
                    que_type = np.array(int(que_type))
                    que_type_list.append(que_type)
                # load option embedding
                option = 'a'
                opt_embs = []
                while os.path.exists(os.path.join(ndq_path, option + '.pkl')):
                    with open(os.path.join(ndq_path, option + '.pkl'), 'rb') as f_opt:
                        opt_emb = pickle.load(f_opt).reshape(-1, cfg.init_word_emb)
                        x_dim = opt_emb.shape[0]
                        pad_dim = cfg.max_opt_len - x_dim
                        if pad_dim >= 0:
                            opt_emb = np.pad(opt_emb, ((0, pad_dim), (0, 0)), 'constant', constant_values=(0, 0))
                        else:
                            opt_emb = opt_emb[:pad_dim]
                        opt_embs.append(opt_emb)

                    option = chr(ord(option) + 1)
                difference = cfg.max_opt_count - len(opt_embs)
                # pad the number of options to max_opt_num
                if difference > 0:
                    for i in range(difference):
                        opt_emb = np.zeros((cfg.max_opt_len, cfg.init_word_emb))
                        opt_embs.append(opt_emb)

                opt_list.append(opt_embs)
                # load correct answer
                with open(os.path.join(ndq_path, 'correct_answer.pkl'), 'rb') as f_ans:
                    ans_one_hot = pickle.load(f_ans).reshape(-1, cfg.max_opt_count)
                    ans_list.append(ans_one_hot)

                # load closest sentence embedding
                with open(os.path.join(ndq_path, 'closest_sent.pkl'), 'rb') as f_closest_sent:
                    closest_sent_emb = pickle.load(f_closest_sent)
                    closest_sent_list.append(closest_sent_emb)

    assert len(que_list) == len(opt_list) == len(ans_list) == len(closest_sent_list) == len(
        que_type_list), 'the number of these list is not equal.'

    return torch.from_numpy(np.array(que_list, dtype='float32')), \
           torch.from_numpy(np.array(opt_list, dtype='float32')), \
           torch.from_numpy(np.array(ans_list, dtype='float32')), \
           torch.from_numpy(np.array(closest_sent_list, dtype='float32')), \
           torch.from_numpy(np.array(que_type_list, dtype='float32'))


def load_diagram_data(cfg):
    dd_matrix_list = []
    dd_node_emb_list = []
    que_list = []
    dq_matrix_list = []
    dq_node_emb_list = []
    opt_list = []
    ans_list = []
    closest_sent_list = []
    count_dict = collections.defaultdict(int)

    if cfg.mode == 'train':
        if cfg.splits == 'train+val':
            slice_paths = ['train', 'val']
            print('Note: begin to load training set')
        else:
            slice_paths = ['train', 'test']
            print('Note: begin to load training set')
    else:
        if cfg.splits == 'train+val':
            slice_paths = ['test']
            print('Note: begin to load test set')
        else:
            slice_paths = ['val']
            print('Note: begin to load test set')

    for slice_path in slice_paths:
        path = cfg.pre_path + slice_path + cfg.suf_path
        lesson_list = get_list_of_dirs(path)

        for lesson in lesson_list:
            dd_list_path = os.path.join(path, lesson)
            dd_list = get_dd_list_of_dirs(dd_list_path)
            dd_matrices = []
            dd_node_embs = []
            for i, dd in enumerate(dd_list):
                if i < cfg.max_dd_num:
                    dd_path = os.path.join(dd_list_path, dd)
                    # load dd
                    with open(os.path.join(dd_path, 'adjacency_matrix_diagram.pkl'), 'rb') as f_amd:
                        amd = pickle.load(f_amd)
                        dim = amd.shape[0]
                        pad_dim = cfg.max_diagram_node - dim

                        if pad_dim >= 0:
                            amd = np.pad(amd, ((0, pad_dim), (0, pad_dim)), 'constant', constant_values=(0, 0))
                        else:
                            amd = amd[:pad_dim, :pad_dim]
                        dd_matrices.append(amd)

                    with open(os.path.join(dd_path, 'node_embedding.pkl'), 'rb') as f_node_emb:
                        node_emb = pickle.load(f_node_emb)
                        expand_node_emb = np.zeros((cfg.max_diagram_node, cfg.init_word_emb))
                        for i, emb in enumerate(node_emb.values()):
                            if i < cfg.max_diagram_node:
                                expand_node_emb[i, :] = emb
                            else:
                                break
                        dd_node_embs.append(expand_node_emb)

            difference = cfg.max_dd_num - len(dd_matrices)
            if difference > 0:
                for i in range(difference):
                    dd_matrix = np.zeros((cfg.max_diagram_node, cfg.max_diagram_node))
                    dd_node_emb = np.zeros((cfg.max_diagram_node, cfg.init_word_emb))
                    dd_matrices.append(dd_matrix)
                    dd_node_embs.append(dd_node_emb)

            dq_list_path = os.path.join(path, lesson)
            dq_list = get_dq_list_of_dirs(dq_list_path)

            assert len(dd_matrices) == len(
                dd_node_embs) == cfg.max_dd_num, 'the length of these two list is not equal with max dd num.'

            for dq in dq_list:
                dd_matrix_list.append(dd_matrices)
                dd_node_emb_list.append(dd_node_embs)
                dq_path = os.path.join(dq_list_path, dq)
                # load question.pkl
                with open(os.path.join(dq_path, 'Question.pkl'), 'rb') as f_que:
                    que_emb = pickle.load(f_que).reshape(-1, cfg.init_word_emb)
                    x_dim = que_emb.shape[0]
                    pad_dim = cfg.max_que_len - x_dim
                    if pad_dim >= 0:
                        que_emb = np.pad(que_emb, ((0, pad_dim), (0, 0)), 'constant', constant_values=(0, 0))
                    else:
                        que_emb = que_emb[:pad_dim]
                    que_list.append(que_emb)

                with open(os.path.join(dq_path, 'adjacency_matrix_diagram.pkl'), 'rb') as f_amd:
                    amd = pickle.load(f_amd)
                    dim = amd.shape[0]
                    pad_dim = cfg.max_diagram_node - dim
                    if pad_dim >= 0:
                        amd = np.pad(amd, ((0, pad_dim), (0, pad_dim)), 'constant', constant_values=(0, 0))
                    else:
                        amd = amd[:pad_dim, :pad_dim]
                    if count_dict[np.sum(amd)] != 0:
                        count_dict[np.sum(amd)] += 1
                    else:
                        count_dict[np.sum(amd)] = 1
                    dq_matrix_list.append(amd)

                with open(os.path.join(dq_path, 'node_embedding.pkl'), 'rb') as f_node_emb:
                    node_emb = pickle.load(f_node_emb)
                    expand_node_emb = np.zeros((cfg.max_diagram_node, cfg.init_word_emb))
                    for i, emb in enumerate(node_emb.values()):
                        if i < cfg.max_diagram_node:
                            expand_node_emb[i, :] = emb
                        else:
                            break
                    dq_node_emb_list.append(expand_node_emb)
                # load option embedding
                option = 'a'
                opt_embs = []
                while os.path.exists(os.path.join(dq_path, option + '.pkl')):
                    with open(os.path.join(dq_path, option + '.pkl'), 'rb') as f_opt:
                        opt_emb = pickle.load(f_opt).reshape(-1, cfg.init_word_emb)
                        x_dim = opt_emb.shape[0]
                        pad_dim = cfg.max_opt_len - x_dim
                        if pad_dim >= 0:
                            opt_emb = np.pad(opt_emb, ((0, pad_dim), (0, 0)), 'constant', constant_values=(0, 0))
                        else:
                            opt_emb = opt_emb[:pad_dim]
                        opt_embs.append(opt_emb)
                    option = chr(ord(option) + 1)
                difference = cfg.max_opt_count - len(opt_embs)
                # pad the number of options to max_opt_num
                if difference > 0:
                    for i in range(difference):
                        opt_emb = np.zeros((cfg.max_opt_len, cfg.init_word_emb))
                        opt_embs.append(opt_emb)

                opt_list.append(opt_embs)
                # load correct answer
                with open(os.path.join(dq_path, 'correct_answer.pkl'), 'rb') as f_ans:
                    ans_one_hot = pickle.load(f_ans).reshape(1, -1)
                    ans_one_hot = ans_one_hot[:, :cfg.max_opt_count]
                    ans_list.append(ans_one_hot)

                # load closest sentence embedding
                with open(os.path.join(dq_path, 'closest_sent.pkl'), 'rb') as f_closest_sent:
                    closest_sent_emb = pickle.load(f_closest_sent)
                    closest_sent_list.append(closest_sent_emb)

    assert len(que_list) == len(dq_matrix_list) == len(dq_node_emb_list) == len(dd_matrix_list) == len(
        dd_node_emb_list) == len(opt_list) == len(ans_list) == len(
        closest_sent_list), 'the number of these list is not equal.'
    # count_dict = sorted(count_dict.items(), key=lambda x: x[0])
    # print(count_dict)
    # plt.bar(range(len(count_dict)), list(count_dict.keys()), align='center')
    # plt.xticks(range(len(count_dict)), list(count_dict.values()))
    # plt.xticks(fontsize=1)
    # plt.savefig("mygraph" + str(slice_paths) + '.png', dpi=1200)
    # plt.show()

    return torch.from_numpy(np.array(que_list, dtype='float32')), \
           torch.from_numpy(np.array(opt_list, dtype='float32')), \
           torch.from_numpy(np.array(dq_matrix_list, dtype='float32')), \
           torch.from_numpy(np.array(dq_node_emb_list, dtype='float32')), \
           torch.from_numpy(np.array(dd_matrix_list, dtype='float32')), \
           torch.from_numpy(np.array(dd_node_emb_list, dtype='float32')), \
           torch.from_numpy(np.array(ans_list, dtype='float32')), \
           torch.from_numpy(np.array(closest_sent_list, dtype='float32'))


def count_accurate_prediction_text(label_ix, pred_ix, qt_iter):
    result = label_ix.eq(pred_ix).cpu()
    qt_tf = 0
    qt_mc = 0

    tf_sum = 0
    mc_sum = 0

    for i, flag in enumerate(result):
        if flag == True:
            if qt_iter[i] == 0:
                qt_tf += 1
                tf_sum += 1
            else:
                qt_mc += 1
                mc_sum += 1
        else:
            if qt_iter[i] == 0:
                tf_sum += 1
            else:
                mc_sum += 1
    return qt_tf, qt_mc, tf_sum, mc_sum
