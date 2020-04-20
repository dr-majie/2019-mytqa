import os
import shutil

import numpy as np
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
import string
from preprocessing.read_json import read_json
from preprocessing.build_graph import build_diagram_graph, get_anchor_nodes_of_que, get_anchor_nodes_of_all, \
    get_dependency_parsing, detect_exception, convert_num2words
import torch
from stanfordcorenlp import StanfordCoreNLP
import torch.nn.functional as F
import collections
import copy


class generate_network_ready_files():
    def __init__(self, word2vec_path, processed_data_path, is_test_data, word_vec_size, max_q_length, max_option_length,
                 max_opt_count, max_sent_para, max_words_sent, op_path=None):
        self.processed_data_path = processed_data_path
        self.raw_text_path = os.path.join(processed_data_path, 'text_question_sep_files')
        self.is_test_data = is_test_data
        self.word2vec_path = word2vec_path

        # self.raw_diagram_path = os.path.join(processed_data_path, 'graph_files')

        if not os.path.exists(self.raw_text_path):
            read_json_data = read_json(os.path.dirname(processed_data_path), is_test_data)
            read_json_data.read_json_do_sanity_create_closest_sent(self.word2vec_path)

        if op_path is None:
            op_path = os.path.join(processed_data_path, 'one_hot_files')

        if not os.path.exists(op_path):
            os.makedirs(op_path)

        self.op_path = op_path
        self.word_vec_size = word_vec_size
        self.num_of_words_in_opt = max_option_length
        self.num_of_words_in_question = max_q_length
        self.num_of_sents_in_closest_para = max_sent_para
        self.num_of_words_in_sent = max_words_sent
        self.num_of_words_in_closest_sentence = max_sent_para * max_words_sent
        self.num_of_options_for_quest = max_opt_count
        self.lessons_list = self.get_list_of_dirs(self.raw_text_path)
        self.unknown_words_vec_dict = None
        self.unknown_words_vec_dict_file = 'unk_word2vec_dict.pkl'
        self.common_files_path = '../common_files'

        if not os.path.exists(self.common_files_path):
            os.makedirs(self.common_files_path)

    def get_list_of_dirs(self, dir_path):
        dirlist = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
        dirlist.sort()
        return dirlist

    # def get_list_of_files(self, file_path, file_extension='.txt'):
    #     filelist = []
    #     for root, dirs, files in os.walk(file_path):
    #         for filen in files:
    #             if filen.endswith(file_extension):
    #                 filelist.append(filen)
    #     filelist.sort()
    #     return filelist
    def get_list_of_files(self, file_path, file_extension='png'):
        filelist = []
        for root, dirs, files in os.walk(file_path):
            for filen in files:
                if not filen.endswith(file_extension):
                    filelist.append(filen)
        filelist.sort()
        return filelist

    def handle_unknown_words(self, word):
        fname = self.unknown_words_vec_dict_file

        if self.unknown_words_vec_dict is None:
            print('Dict is none')

            if os.path.isfile(os.path.join(self.common_files_path, fname)):
                print('Dict file exist')

                with open(os.path.join(self.common_files_path, fname), 'rb') as f:
                    self.unknown_words_vec_dict = pickle.load(f)
            else:
                print('Dict file does not exist')
                self.unknown_words_vec_dict = {}

        if self.unknown_words_vec_dict.get(word, None) is not None:
            # print('word present in dictionary : ', word)
            vec = self.unknown_words_vec_dict.get(word, None)
        else:
            #print('word is not present in dictionary : ', word)
            vec = np.random.rand(1, self.word_vec_size)
            self.unknown_words_vec_dict[word] = vec
        return vec

    def get_vec_for_word(self, model, word):
        try:
            vec = model[word]
            return vec
        except:
            #print('Vector not in model for word: ', word)
            vec = self.handle_unknown_words(word)
            return vec

    def write_vecs_to_file(self, model, raw_data_content, word2vec_file, is_correct_answer_file=False,
                           is_closest_para_file=False):
        all_vec_array = np.array([])
        number_of_words = 0
        break_loop = False

        if is_correct_answer_file:
            word = raw_data_content[0].strip().lower()
            pos = ord(word) - 97
            all_vec_array = 0 * np.ones(self.num_of_options_for_quest)
            all_vec_array[pos] = 1

        elif is_closest_para_file:
            all_vec_array = np.zeros((self.num_of_sents_in_closest_para, self.num_of_words_in_sent, self.word_vec_size))
            sents = sent_tokenize(raw_data_content)

            for i in range(len(sents)):
                words = word_tokenize(sents[i])
                words = [w for w in words if w not in string.punctuation]
                # sanity check
                if len(words) > self.num_of_words_in_sent:
                    words = words[:self.num_of_words_in_sent]

                for j in range(len(words)):
                    word = words[j].strip().lower()
                    vec = self.get_vec_for_word(model, word)
                    all_vec_array[i, j, :] = vec

        else:
            for sent in raw_data_content:
                words = word_tokenize(sent)
                words = [w for w in words if w not in string.punctuation]  ## to remove punctuations

                for word in words:
                    word = word.strip().lower()
                    vec = self.get_vec_for_word(model, word)
                    all_vec_array = np.append(all_vec_array, vec)
                    number_of_words += 1

                    if number_of_words > self.num_of_words_in_closest_sentence - 1:
                        break_loop = True
                        break

                if break_loop:
                    break

        pickle.dump(all_vec_array, word2vec_file)
        word2vec_file.close()

    def build_textual_graph(self, que_path, graph_que_ins_path, model, scp):
        anchor_nodes_of_que = get_anchor_nodes_of_que(que_path)
        anchor_nodes_all = get_anchor_nodes_of_all(que_path, anchor_nodes_of_que)
        dependency_trees = get_dependency_parsing(os.path.join(que_path, 'closest_sent.txt'), scp)
        option = 'a'
        node_count = []
        for anchor_nodes in anchor_nodes_all:
            node_of_graph = set()
            relation = set()
            # building graph (que, option)
            # for depth in range(2):
            #     for node in anchor_nodes:
            #         for tree in dependency_trees:
            #             for edge in tree:
            #                 if node in edge:
            #                     relation.add(edge)
            #                     node_of_graph.add(edge[1])
            #                     node_of_graph.add(edge[2])
            #     anchor_nodes = set()
            #     anchor_nodes.update(node_of_graph)
            initial_anchor_nodes = copy.deepcopy(anchor_nodes)
            for depth in range(2):
                for node in initial_anchor_nodes:
                    for tree in dependency_trees:
                        for edge in tree:
                            if node in edge:
                                relation.add(edge)
                                node_of_graph.add(edge[1])
                                node_of_graph.add(edge[2])
                initial_anchor_nodes = set()
                initial_anchor_nodes.update(node_of_graph)

            # size = len(node_of_graph)
            # adjacency_matrix = np.zeros((size, size))
            node_dict = collections.OrderedDict()

            if not node_of_graph:
                size = len(anchor_nodes)
                for node in anchor_nodes:
                    node_of_graph.add(node)
                for i, node in enumerate(node_of_graph):
                    node_dict[node] = i
                adjacency_matrix = np.ones((size, size))
            else:
                size = len(node_of_graph)
                adjacency_matrix = np.zeros((size, size))
                # node_dict = collections.OrderedDict()

                for i, node in enumerate(node_of_graph):
                    node_dict[node] = i

                for edge in relation:
                    adjacency_matrix[node_dict[edge[1]]][node_dict[edge[2]]] = 1
                    adjacency_matrix[node_dict[edge[2]]][node_dict[edge[1]]] = 1

            with open(os.path.join(graph_que_ins_path, 'adjacency_matrix_' + option + '.pkl'), 'wb') as f_graph:
                pickle.dump(adjacency_matrix, f_graph)

            for node in node_dict:
                node_dict[node] = self.get_vec_for_word(model, node)

            with open(os.path.join(graph_que_ins_path, 'node_embedding_' + option + '.pkl'), 'wb') as f_node_emb:
                pickle.dump(node_dict, f_node_emb)

            node_count.append(len(node_dict))
            option = chr(ord(option) + 1)
        print(node_count)

    def write_diagram_vecs_to_file(self, model, node_dict, adjacency_matrix, graph_que_ins_path):
        if node_dict:
            for node in node_dict:
                # if len(node) == 1:
                if len(node.split(" ")) == 1:
                    node_dict[node] = self.get_vec_for_word(model, node)
                else:
                    vec_arr = []
                    words = word_tokenize(node)
                    words = [w for w in words if w not in string.punctuation]
                    for word in words:
                        vec_arr.append(self.get_vec_for_word(model, word).reshape(300, ))
                    vec_sum = np.array(vec_arr).sum(axis=1)
                    vec_sum_input = torch.from_numpy(np.array([vec_sum]))
                    att_vec = F.softmax(vec_sum_input, dim=1)
                    vec_arr = torch.from_numpy(np.array(vec_arr))
                    weighted_vec = torch.mm(att_vec, vec_arr)
                    node_dict[node] = weighted_vec
        else:
            node_dict = {"no_node": np.zeros((1, 300))}
            adjacency_matrix = np.array([[0]])

        with open(os.path.join(graph_que_ins_path, 'node_embedding.pkl'), 'wb') as f_node_emb:
            pickle.dump(node_dict, f_node_emb)
        with open(os.path.join(graph_que_ins_path, 'adjacency_matrix_diagram' + '.pkl'), 'wb') as f_graph:
            pickle.dump(adjacency_matrix, f_graph)

    def generate_word2vec_for_all(self,scp):

        print(20 * '*')
        print('GENERATING NETWORK READY FILES.')
        model = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)
        # scp = StanfordCoreNLP(r'/data/kf/majie/stanford-corenlp-full-2018-10-05/')

        max_nodes_of_lessons = []
        min_nodes_of_lessons = []
        for lesson in self.lessons_list:
            node_count_dq = []
            node_count_dd = []

            l_dir = os.path.join(self.raw_text_path, lesson)
            print('Lesson : ', lesson)
            op_l_dir = os.path.join(self.op_path, lesson)

            if not os.path.exists(op_l_dir):
                os.makedirs(op_l_dir)

            questions_dir = self.get_list_of_dirs(l_dir)
            questions_dir = [name for name in questions_dir]
            for question_dir in questions_dir:
                print('Question : ', question_dir)
                if not os.path.exists(os.path.join(op_l_dir, question_dir)):
                    os.makedirs(os.path.join(op_l_dir, question_dir))
                if question_dir.startswith("NDQ") or question_dir.startswith("DQ"):
                    file_list = self.get_list_of_files(os.path.join(l_dir, question_dir))
                    # print('Question : ', question_dir)
                    for fname in file_list:
                        if fname == 'correct_answer.txt':
                            is_correct_answer_file = True
                        else:
                            is_correct_answer_file = False
                        if fname == 'question_type.pkl':
                            old_path = os.path.join(self.raw_text_path, lesson, question_dir, fname)
                            new_path = os.path.join(op_l_dir, question_dir)
                            shutil.copy2(old_path, new_path)
                        elif fname == 'coordinate.txt':
                            pass
                        else:
                            with open(os.path.join(l_dir, question_dir, fname), 'r') as f:
                                if fname == 'closest_sent.txt':
                                    is_closest_para_file = True
                                    try:
                                        text = f.readlines()[0]
                                        raw_data_content = ''
                                        count = 0
                                        for s in sent_tokenize(text):
                                            if len(s.split()) > self.num_of_words_in_sent:
                                                raw_data_content += ' '.join(s.split()[:self.num_of_words_in_sent])
                                                raw_data_content += '. '
                                            else:
                                                raw_data_content += ' '.join(s.split())
                                                raw_data_content += ' '

                                            count += 1
                                            if count == self.num_of_sents_in_closest_para:
                                                break
                                    except:
                                        raw_data_content = f.readlines()
                                else:
                                    is_closest_para_file = False
                                    raw_data_content = f.readlines()
                        # if fname == 'coordinate.txt':
                        #     pass
                        # elif fname == 'question_type.pkl':
                        #     old_path = os.path.join(self.raw_text_path, lesson, question_dir, fname)
                        #     new_path = os.path.join(op_l_dir, question_dir)
                        #     shutil.copy2(old_path, new_path)

                        # else:
                            f = open(os.path.join(op_l_dir, question_dir, fname[:-4] + '.pkl'), 'wb')
                            self.write_vecs_to_file(model, raw_data_content, f, is_correct_answer_file,
                                                    is_closest_para_file)
                            f.close()

                que_ins_path = os.path.join(l_dir, question_dir)
                graph_que_ins_path = os.path.join(op_l_dir, question_dir)
                if question_dir.startswith('DQ'):
                    node_dict, adjacency_matrix = build_diagram_graph(que_ins_path, 'DQ', scp)
                    self.write_diagram_vecs_to_file(model, node_dict, adjacency_matrix, graph_que_ins_path)
                    self.build_textual_graph(que_ins_path, graph_que_ins_path, model, scp)
                    node_count_dq.append(len(node_dict))
                elif question_dir.startswith('DD'):
                    node_dict, adjacency_matrix = build_diagram_graph(que_ins_path, 'DD', scp)
                    self.write_diagram_vecs_to_file(model, node_dict, adjacency_matrix, graph_que_ins_path)
                    node_count_dd.append(len(node_dict))
                else:
                    self.build_textual_graph(que_ins_path, graph_que_ins_path, model, scp)

            # if len(node_count_dq) != 0 or len(node_count_dd) != 0:
            #     if len(node_count_dq) != 0:
            #         print("count of dq nodes:", node_count_dq)
            #         print("count of DQ:", len(node_count_dq))
            #         print("max count of dq nodes:", max(node_count_dq))
            #         max_nodes_of_lessons.append(max(node_count_dq))
            #         print("min count of dq nodes:", min(node_count_dq))
            #         min_nodes_of_lessons.append(min(node_count_dq))
            #     if len(node_count_dd) != 0:
            #         print("count of dd nodes:", node_count_dd)
            #         print("count of DD:", len(node_count_dd))
            #         print("max count of dd nodes:", max(node_count_dd))
            #         max_nodes_of_lessons.append(max(node_count_dd))
            #         print("min count of dd nodes:", min(node_count_dd))
            #         min_nodes_of_lessons.append(min(node_count_dd))
            # else:
            #     print("the lesson only has NDQ")

            print(20 * '***')
        # scp.close()
        # print("max count of nodes in all lessons:", max(max_nodes_of_lessons))
        # print("min count of nodes in all lessons:", min(min_nodes_of_lessons))

        print('saving final unknown word2vec dictionary to file')
        f = open(os.path.join(self.common_files_path, self.unknown_words_vec_dict_file), 'wb')
        pickle.dump(self.unknown_words_vec_dict, f)
        f.close()
