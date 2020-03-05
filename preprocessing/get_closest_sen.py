from preprocessing.query_expansion import sentence_retriever_using_w2vec
import os
import pickle
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors


class get_closest_sentences():
    def __init__(self, processed_data_path):
        self.processed_data_path = processed_data_path
        self.raw_text_path = os.path.join(processed_data_path, 'text_question_sep_files')
        self.lessons_list = self.get_list_of_dirs(self.raw_text_path)

    def get_list_of_dirs(self, dir_path):
        dirlist = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
        dirlist.sort()
        return dirlist

    def get_list_of_files(self, file_path, file_extension='.txt'):
        filelist = [name for name in os.listdir(file_path) if
                    name.endswith(file_extension) and not os.path.isdir(os.path.join(file_path, name))]
        filelist.sort()
        return filelist

    def convert_list_to_string(self, ip_list):
        op_string = ''
        for sent in ip_list:
            op_string += ' ' + sent.strip()
        return op_string

    def get_query_based_sentences(self, topic_content, question_content, sent_f_handle):
        sent_retr = sentence_retriever_using_w2vec(self.W2V_MODEL)
        closest_sentences = sent_retr.get_related_sentences(topic_content,
                                                            self.convert_list_to_string(question_content))
        sent_f_handle.write(closest_sentences)

    def generate_closest_sentence(self, word2vec_path):

        w2vec_path = word2vec_path
        self.W2V_MODEL = KeyedVectors.load_word2vec_format(w2vec_path, binary=True, unicode_errors='replace')
        self.W2V_MODEL.init_sims(replace=True)

        print(20 * '*')
        print('GENERATING CLOSEST SENTENCE')

        topic_fname = 'topics.txt'
        question_fname = 'Question.txt'
        f_ext = '.txt'
        sent_closest_to_question_fname = 'closest_sent.txt'

        for lesson in self.lessons_list:
            print('lesson : ', lesson)
            l_dir = os.path.join(self.raw_text_path, lesson)

            with open(os.path.join(l_dir, topic_fname), 'r') as f:
                topic_content = f.read()

            topic_content = topic_content.split('\n')
            topic_content = [t for t in topic_content if t != '']
            questions_dir = self.get_list_of_dirs(l_dir)

            for question_dir in questions_dir:
                if question_dir.startswith('NDQ'):
                    print('Question : ', question_dir)
                    with open(os.path.join(l_dir, question_dir, question_fname), 'r') as f:
                        question_content = f.readlines()

                    option = 'a'
                    while os.path.exists(os.path.join(l_dir, question_dir, option + f_ext)):
                        with open(os.path.join(l_dir, question_dir, option + f_ext), 'r') as f:
                            opt = f.readlines()
                        question_content.append(self.convert_list_to_string(opt))
                        option = chr(ord(option) + 1)

                    sent_f_handle = open(os.path.join(l_dir, question_dir, sent_closest_to_question_fname), 'w')
                    self.get_query_based_sentences(topic_content, question_content, sent_f_handle)
                    sent_f_handle.close()
