# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/3/21 16:04
# @Author:Ma Jie
# @FileName: test.py
# -----------------------------------------------
import pickle
from model.rafr_tqa.gat import GAT
import torch
import numpy as np

"""
np.set_printoptions(threshold=np.inf)
adjacency_path = '/data/kf/majie/wangyaxian/2019-tqa/data/train/processed_data/graph_files/L_0002/NDQ_000046/a.pkl'
emb_path = '/data/kf/majie/wangyaxian/2019-tqa/data/train/processed_data/graph_files/L_0002/NDQ_000046/embedding_a.pkl'

f_adjacency = open(adjacency_path, 'rb')
matrix = pickle.load(f_adjacency)
identify_matrix = np.identity(matrix.shape[0])
matrix = np.add(matrix, identify_matrix)
matrix = torch.tensor(matrix)

f_emb = open(emb_path, 'rb')
emb = pickle.load(f_emb)

emb_np = np.zeros((len(emb), 300))
for i, vec in enumerate(emb.values()):
    emb_np[i:] = vec

model = GAT(emb_np.shape[1], 300, 300, 0., 0.2, 8)
out = model(torch.FloatTensor(emb_np), matrix)
out = out.detach().numpy()
"""
test = 'A.5,10,20,30 B.5,10,15,20 C.10,20,30 D.5'
test = test.strip().split()
print(test)
