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
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import string
import copy

def get_list_of_dirs(dir_path):
    dirlist = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    dirlist.sort()
    return dirlist


def detect_exception(tup, words):
    if tup[1] > len(words) or tup[2] > len(words):
        return False

    if 'ROOT' in tup:
        return False

    if 'punct' in tup:
        return False

    if words[tup[1] - 1] in str(string.punctuation) or words[tup[2] - 1] in str(string.punctuation):
        return False
    # if words[tup[1] - 1] in string.punctuation or words[tup[2] - 1] in string.punctuation:
    #     return False
    return True


def convert_num2words(tuple_list, words):
    # for tup in tuple_list:
    #     print(tup)
    return [(tup[0], words[tup[1] - 1].lower(), words[tup[2] - 1].lower()) for tup in tuple_list if
            detect_exception(tup, words)]


def get_dependency_parsing(closest_sent_path, scp):
    dependency_trees = []
    with open(closest_sent_path, 'r') as f_closest_sent:
        closest_sents = f_closest_sent.readlines()[0]
        closest_sents = sent_tokenize(closest_sents)
        for sent in closest_sents:
            sent = sent.replace('-', ' ')
            tree = scp.dependency_parse(sent)
            words = word_tokenize(sent)
            tree = convert_num2words(tree, words)
            dependency_trees.append(tree)
    return dependency_trees


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


def build_diagram_graph(que_path, diagram_type):
    scp = StanfordCoreNLP(r'/data/kf/majie/stanford-corenlp-full-2018-10-05/')
    if diagram_type == 'DQ':
        diagram_info, nodes_of_diagram = get_info_of_diagram_DQ(que_path)
        dependency_trees = get_dependency_parsing(os.path.join(que_path, 'closest_sent.txt'), scp)
    else:
        diagram_info, nodes_of_diagram = get_info_of_diagram_DD(que_path)
        file_names = os.listdir(que_path + '/')
        image_info_url = [os.path.join(que_path + '/', file) for file in file_names if file.endswith(".txt")]
        with open(image_info_url[0], 'r') as load_f:
            image_info = json.load(load_f)
            text_info = image_info['dd_text']
            text_sents = sent_tokenize(text_info)
        dependency_trees = []
        for sent in text_sents:
            sent = sent.replace('-', ' ')
            tree = scp.dependency_parse(sent)
            words = word_tokenize(sent)
            tree = convert_num2words(tree, words)
            dependency_trees.append(tree)
    scp.close()
    img_name = [name for name in os.listdir(que_path) if name.endswith(".png")]
    img_url = os.path.join(que_path, img_name[0])
    image = Image.open(img_url)
    size = len(diagram_info)
    adjacency_matrix = np.zeros((size, size))
    node_of_diagram_graph = set()
    # relation = set()
    node_dict = collections.OrderedDict()

    all_dependency_relations = set()
    for tree in dependency_trees:
        dependency_relations = set()
        for edge in tree:
            edge = (edge[1], edge[2])
            dependency_relations.add(edge)

        for depth in range(2):
            single_dependency_relations = copy.deepcopy(dependency_relations)
            for edgei in single_dependency_relations:
                for edgej in single_dependency_relations:
                    if edgei[1] == edgej[0]:
                        edge = (edgei[0], edgej[1])
                        dependency_relations.add(edge)
        for relation in dependency_relations:
            all_dependency_relations.add(relation)

    relation = set()
    count_of_relations_in_dependency = 0
    count_of_relations_in_location = 0
    flag = 0
    for i in range(size):
        for j in range(size):
            if i != j:
                node_s = diagram_info[i]['WordText']
                node_t = diagram_info[j]['WordText']
                # for tree in dependency_trees:
                #     for edge in tree:
                for edge in all_dependency_relations:
                    if node_s in edge and node_t in edge:
                        flag = 1
                        node_of_diagram_graph.add(node_s)
                        node_of_diagram_graph.add(node_t)
                        edge_of_diagram = (i, j)
                        relation.add(edge_of_diagram)
                        count_of_relations_in_dependency += 1
                    elif (edge[0] in node_s and edge[1] in node_t) or (edge[1] in node_s and edge[0] in node_t):
                        flag = 1
                        node_of_diagram_graph.add(node_s)
                        node_of_diagram_graph.add(node_t)
                        edge_of_diagram = (i, j)
                        relation.add(edge_of_diagram)
                        count_of_relations_in_dependency += 1
                    else:
                        pass

                if flag == 0:
                    xi_axis = diagram_info[i]['Coordinate']['Center'][0]
                    yi_axis = diagram_info[i]['Coordinate']['Center'][1]
                    xj_axis = diagram_info[j]['Coordinate']['Center'][0]
                    yj_axis = diagram_info[j]['Coordinate']['Center'][1]
                    if max(abs(xi_axis - xj_axis) / image.size[0], abs(yi_axis - yj_axis) / image.size[1]) < 0.3:
                        node_of_diagram_graph.add(diagram_info[i]['WordText'])
                        node_of_diagram_graph.add(diagram_info[j]['WordText'])
                        edge_of_diagram = (i, j)
                        relation.add(edge_of_diagram)
                        count_of_relations_in_location += 1

    print("count find by dependency relations", count_of_relations_in_dependency)
    print("count find by location relations", count_of_relations_in_location)
    print("count of all relations in diagram", len(relation))
    # scp.close()
    for edge in relation:
        adjacency_matrix[edge[0]][edge[1]] = 1
        adjacency_matrix[edge[1]][edge[0]] = 1

    for i, node in enumerate(node_of_diagram_graph):
        node_dict[node] = i
    return node_dict, adjacency_matrix


def get_info_of_diagram_DQ(que_path):
    nodes_of_diagram = set()
    with open(os.path.join(que_path, 'coordinate.txt')) as f:
        dic = json.load(f)
        for key in dic.keys():
            diagram_info = dic[key]
            for detailed_info in diagram_info:
                wordtext = detailed_info['WordText']
                nodes_of_diagram.add(wordtext)
    return diagram_info, nodes_of_diagram


def get_info_of_diagram_DD(que_path):
    # image_info_url = [name for name in os.listdir(que_path) if name.endswith(".txt")]
    file_names = os.listdir(que_path + '/')
    image_info_url = [os.path.join(que_path + '/', file) for file in file_names if file.endswith(".txt")]
    with open(image_info_url[0], 'r') as load_f:
        image_info = json.load(load_f)
        diagram_info = image_info['dd_coordinate']
        nodes_of_diagram = set()
        for item in diagram_info:
            wordtext = item['WordText']
            nodes_of_diagram.add(wordtext)
    return diagram_info, nodes_of_diagram
