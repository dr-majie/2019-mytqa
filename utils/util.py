# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/24 20:05
# @Author:Ma Jie
# @FileName: util.py
# -----------------------------------------------
import os, pickle, json, collections, string
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
from shutil import copyfile

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


def make_mask(feature):
    return torch.sum(torch.abs(feature), dim=-1) == 0


def make_mask_num(feature):
    return torch.sum((torch.sum(torch.abs(feature), dim=-1)), dim=-1) == 0


def print_obj(obj):
    for item in obj.__dict__.items():
        print(item)


def load_csdia_data(cfg):
    que_list = []
    opt_list = []
    ans_list = []
    dia_mat_list = []
    dia_nod_list = []

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
            slice_paths = ['test']
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
            slice_paths = ['train']
            print('Note: begin to load training set')
    else:
        if cfg.splits == 'train+val':
            slice_paths = ['test']
            print('Note: begin to load test set')
        else:
            slice_paths = ['test']
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


def process_csdia_data():
    root_path = '/data/majie/majie/codehub/2019-mytqa/processed_csdia_data'
    csdia_diagram = '/data/majie/majie/codehub/2019-mytqa/csdia data/diagram/Queue/'
    csdia_diagram_anno = '/data/majie/majie/codehub/2019-mytqa/csdia data/annotation/Queue/Queue.json'
    qas = '/data/majie/majie/codehub/2019-mytqa/csdia data/QA/Queue/'
    mc_path = os.path.join(root_path, 'mc')
    tf_path = os.path.join(root_path, 'tf')
    max_nodes = 10
    threshold = 0.2

    if not os.path.exists(root_path):
        os.mkdir(root_path)
        os.mkdir(mc_path)
        os.mkdir(tf_path)

    # load the diagram information.
    with open(csdia_diagram_anno, 'r') as f:
        dias_info = json.load(f)

    # load sentence transformer
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    for dia in dias_info['Queue.json']:
        dia_name = dia['filename'].replace('.png', '')
        img = Image.open(os.path.join(csdia_diagram, dia_name + '.png'))
        width, height = img.size
        regions = dia['regions']
        coos = []  # the list of coordinates
        reg_embs = np.zeros((max_nodes, 768))
        adj_mat = np.zeros((max_nodes, max_nodes))  # adjacent matrix
        dia_mc_path = os.path.join(mc_path, dia_name)
        dia_tf_path = os.path.join(tf_path, dia_name)

        # make dia_name file
        if not os.path.exists(dia_mc_path):
            os.mkdir(dia_mc_path)
        if not os.path.exists(dia_tf_path):
            os.mkdir(dia_tf_path)

        for i, reg in enumerate(regions):
            cent_coo_x = reg['shape_attributes']['x'] + reg['shape_attributes']['width'] / 2.0  # the center coordinate
            cent_coo_y = reg['shape_attributes']['y'] + reg['shape_attributes']['height'] / 2.0
            coos.append((cent_coo_x, cent_coo_y))
            reg_des = reg['region_attributes']['Description']  # the description of regions
            reg_emb = model.encode(reg_des).reshape(1, 768)
            reg_embs[i] = reg_emb
            if i + 1 >= max_nodes:
                break

        # build relation graph
        coos_len = len(coos)
        for i in range(coos_len):
            for j in range(i, coos_len):
                x_diff = abs(coos[i][0] - coos[j][0]) / width
                y_diff = abs(coos[i][1] - coos[j][1]) / width
                if max(x_diff, y_diff) < threshold:
                    adj_mat[i][j] = adj_mat[j][i] = 1

        # read questions corresponding to queue_x.png
        ques = []
        opts = []
        correct_ans = []
        if not os.path.exists(os.path.join(qas, dia_name + '.txt')):
            print(dia_name)
            os.removedirs(os.path.join(root_path, 'mc', dia_name))
            os.removedirs(os.path.join(root_path, 'tf', dia_name))
            continue
        with open(os.path.join(qas, dia_name + '.txt')) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i % 2 == 0:
                    que = line[2:].strip().strip(string.punctuation)
                    ques.append(que)
                else:
                    opt = line.strip().split()
                    opts.append(opt)
                if i + 1 == len(lines):
                    correct_ans.append(line.replace(',', ' ').split(' '))
        for i, opt in enumerate(opts):
            if len(opt) != 2:
                que_emb = model.encode(ques[i]).reshape(1, 768)
                que_path = os.path.join(dia_mc_path, str(i))
                if not os.path.exists(que_path):
                    os.mkdir(que_path)
                answer = np.zeros((1, 4))
                answer[0][ord(correct_ans[0][i]) - 65] = 1
                copyfile(os.path.join(csdia_diagram, dia_name + '.png'), os.path.join(que_path, dia_name + '.png'))
                with open(os.path.join(que_path, 'question.pkl'), 'wb+') as f:
                    pickle.dump(que_emb, f)
                with open(os.path.join(que_path, 'adjacent_matrx.pkl'), 'wb+') as f:
                    pickle.dump(adj_mat, f)
                with open(os.path.join(que_path, 'node_emb.pkl'), 'wb+') as f:
                    pickle.dump(reg_embs, f)
                with open(os.path.join(que_path, 'answer.pkl'), 'wb+') as f:
                    pickle.dump(answer, f)

                o_name = 'a'
                for j, o in enumerate(opt):
                    o = o[2:]
                    o_emb = model.encode(o).reshape(1, 768)
                    with open(os.path.join(dia_mc_path, str(i), o_name + '.pkl'), 'wb+') as f:
                        pickle.dump(o_emb, f)
                    o_name = chr(ord(o_name) + 1)
            else:
                que_emb = model.encode(ques[i]).reshape(1, 768)
                que_path = os.path.join(dia_tf_path, str(i))
                if not os.path.exists(que_path):
                    os.mkdir(que_path)
                answer = np.zeros((1, 2))
                if correct_ans[0][i].lower() in 'true':
                    answer[0][0] = 1
                else:
                    answer[0][1] = 1
                copyfile(os.path.join(csdia_diagram, dia_name + '.png'), os.path.join(que_path, dia_name + '.png'))
                with open(os.path.join(que_path, 'question.pkl'), 'wb+') as f:
                    pickle.dump(que_emb, f)
                with open(os.path.join(que_path, 'adjacent_matrx.pkl'), 'wb+') as f:
                    pickle.dump(adj_mat, f)
                with open(os.path.join(que_path, 'node_emb.pkl'), 'wb+') as f:
                    pickle.dump(reg_embs, f)
                with open(os.path.join(que_path, 'answer.pkl'), 'wb+') as f:
                    pickle.dump(answer, f)

                o_name = 'a'
                for j, o in enumerate(opt):
                    o_emb = model.encode(o).reshape(1, 768)
                    with open(os.path.join(dia_tf_path, str(i), o_name + '.pkl'), 'wb+') as f:
                        pickle.dump(o_emb, f)
                    o_name = chr(ord(o_name) + 1)


if __name__ == '__main__':
    process_csdia_data()
