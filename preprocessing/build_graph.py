# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/15 11:44
# @Author:Ma Jie
# @FileName: build_graph.py
# -----------------------------------------------
import os
import string
import numpy as np
import collections
import pickle
from gensim.models import KeyedVectors
from stanfordcorenlp import StanfordCoreNLP
from nltk.tokenize import word_tokenize, sent_tokenize


def get_list_of_dirs(dir_path):
    dirlist = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    dirlist.sort()
    return dirlist


def detect_exception(tup, words):
    if tup[1] > len(words) or tup[2] > len(words):
        return False

    if 'ROOT' in tup:
        return False

    if words[tup[1] - 1] in str(string.punctuation) or words[tup[2] - 1] in str(string.punctuation):
        return False

    return True

def convert_num2words(tuple_list, words):
    return [(tup[0], words[tup[1] - 1].lower(), words[tup[2] - 1].lower()) for tup in tuple_list if
            detect_exception(tup, words)]


def get_dependency_parsing(closest_sent_path, scp):
    dependency_trees = []

    with open(closest_sent_path, 'r') as f_closest_sent:
        closest_sents = f_closest_sent.readlines()[0]
        closest_sents = sent_tokenize(closest_sents)
        for sent in closest_sents:
            tree = scp.dependency_parse(sent)
            words = word_tokenize(sent)
            tree = convert_num2words(tree, words)
            dependency_trees.append(tree)
    scp.close()
    return dependency_trees


def handle_unknown_words(word):
    unknown_word_file_path = '/data/kf/majie/wangyaxian/2019-tqa/data/common_file/'

    if os.path.exists(unknown_word_file_path):
        with open(os.path.join(unknown_word_file_path, 'unknown_word_vec_dict.pkl'), 'rb') as f_unknown_word:
            unknown_word_vec_dict = pickle.load(f_unknown_word)
    else:
        print('unknown word dictionary is not existing')
        os.makedirs(unknown_word_file_path)
        unknown_word_vec_dict = {}

    if unknown_word_vec_dict.get(word, None) is not None:
        vec = unknown_word_vec_dict.get(word)
    else:
        vec = np.random.rand(1, 300)
        unknown_word_vec_dict[word] = vec

        with open(os.path.join(unknown_word_file_path, 'unknown_word_vec_dict.pkl'), 'wb') as f_unknown_word:
            pickle.dump(unknown_word_vec_dict, f_unknown_word)
    return vec


def get_vec_for_word(model, word):
    try:
        vec = model[word]
        return vec
    except:
        print('Vector not in model for word: ', word)
        vec = handle_unknown_words(word)
        return vec


def build_textual_graph(que_path, graph_que_ins_path, model, scp):
    anchor_nodes_of_que = get_anchor_nodes_of_que(que_path)
    anchor_nodes_all = get_anchor_nodes_of_all(que_path, anchor_nodes_of_que)
    dependency_trees = get_dependency_parsing(os.path.join(que_path, 'closest_sent.txt'), scp)
    option = 'a'

    for anchor_nodes in anchor_nodes_all:
        node_of_graph = set()
        relation = set()
        # building graph (que, option)
        for depth in range(2):
            for node in anchor_nodes:
                for tree in dependency_trees:
                    for edge in tree:
                        if node in edge:
                            relation.add(edge)
                            node_of_graph.add(edge[1])
                            node_of_graph.add(edge[2])
            anchor_nodes = set()
            anchor_nodes.update(node_of_graph)

        size = len(node_of_graph)
        adjacency_matrix = np.zeros((size, size))
        node_dict = collections.OrderedDict()

        for i, node in enumerate(node_of_graph):
            node_dict[node] = i

        for edge in relation:
            adjacency_matrix[node_dict[edge[1]]][node_dict[edge[2]]] = 1
            adjacency_matrix[node_dict[edge[2]]][node_dict[edge[1]]] = 1

        with open(os.path.join(graph_que_ins_path, option + '.pkl'), 'wb') as f_graph:
            pickle.dump(adjacency_matrix, f_graph)
        option = chr(ord(option) + 1)

        for node in node_dict:
            node_dict[node] = get_vec_for_word(model, node)

        with open(os.path.join(graph_que_ins_path, 'node_embedding.pkl'), 'wb') as f_node_emb:
            pickle.dump(node_dict, f_node_emb)


def get_anchor_nodes_of_que(que_path):
    anchor_nodes_of_que = set()
    with open(os.path.join(que_path, 'Question.txt')) as f_que:
        question = f_que.read().translate(str.maketrans('', '', string.punctuation)).lower()
        question = word_tokenize(question)
        for word in question:
            anchor_nodes_of_que.add(word)
    return anchor_nodes_of_que


def get_anchor_nodes_of_all(que_path, anchor_nodes_of_que):
    file_extension = '.txt'
    option = 'a'
    # each set of anchor nodes is consisting of question and option. e.g. {que, option_a}, {que, option_b}
    anchor_nodes_all = []

    while os.path.exists(os.path.join(que_path, option + file_extension)):
        anchor_nodes = set()
        anchor_nodes.update(anchor_nodes_of_que)
        with open(os.path.join(que_path, option + file_extension)) as f_opt:
            option_words = f_opt.read().translate(str.maketrans('', '', string.punctuation)).lower()
            option_words = word_tokenize(option_words)
            for option_word in option_words:
                anchor_nodes.add(option_word)
        anchor_nodes_all.append(anchor_nodes)
        option = chr(ord(option) + 1)
    return anchor_nodes_all


if __name__ == '__main__':
    scp = StanfordCoreNLP(r'/data/kf/majie/stanford-corenlp-full-2018-10-05/')
    slice_path = ['train', 'val', 'test']
    word2vec_path = '/data/kf/majie/wangyaxian/2019-tqa/word2vec/GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    for path in slice_path:
        data_path = '/data/kf/majie/wangyaxian/2019-tqa/data/' + path + '/processed_data/text_question_sep_files/'
        graph_path = '/data/kf/majie/wangyaxian/2019-tqa/data/' + path + '/processed_data/graph_files'
        lesson_list = get_list_of_dirs(data_path)

        for lesson in lesson_list:
            lesson_path = os.path.join(data_path, lesson)
            lesson_graph_path = os.path.join(graph_path, lesson)
            que_ins_list = get_list_of_dirs(lesson_path)

            for que_ins in que_ins_list:
                que_ins_path = os.path.join(lesson_path, que_ins)
                graph_que_ins_path = os.path.join(lesson_graph_path, que_ins)
                if not os.path.exists(graph_que_ins_path):
                    os.makedirs(graph_que_ins_path)

                if que_ins.startswith('DD'):
                    pass
                elif que_ins.startswith('DQ'):
                    pass
                else:
                    build_textual_graph(que_ins_path, graph_que_ins_path, model, scp)
    scp.close()
