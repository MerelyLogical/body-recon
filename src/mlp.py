# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:52:38 2018

@author: zw4215
"""

import numpy as np
from nn import successArray, kNN, mAPNN
from sklearn.neural_network import MLPClassifier
from dataproc import toLabelArray, dataLoad, splitData

def build_sd_sets(t_img_1, t_set):
    t_similar_set = [t_img_2 for t_img_2 in t_set\
        if t_img_2.label == t_img_1.label and t_img_2.camId != t_img_1.camId]
    t_different_set = []
    while (len(t_different_set) != len(t_similar_set)):
        try_idx = np.random.choice(len(t_set), size=1, replace=False)[0]
        if t_set[try_idx].label != t_img_1.label:
            t_different_set.append(t_set[try_idx])
    X = np.zeros((1, 2*len(t_img_1.feature)))
    for t_img_2 in t_similar_set:
        X = np.vstack((X, np.concatenate((t_img_1.feature, t_img_2.feature))))
    for t_img_2 in t_different_set:
        X = np.vstack((X, np.concatenate((t_img_1.feature, t_img_2.feature))))
    X = X[1:,:]
    y = list(np.repeat([0,1], len(t_similar_set)))
    return X, y

def build_mlp_data(t_set):
    X_set = np.zeros((1, 2*len(t_set[0].feature)))
    y_set = []
    for i, t_img in enumerate(t_set):
        if i % 100 == 0:
            print('Building MLP training data... t:{:5d}'.format(i))
        X_img, y_img = build_sd_sets(t_img, t_set)
        X_set = np.vstack((X_set, X_img))
        y_set = y_set + y_img
    return X_set[1:,:], y_set

def build_mlp_test(q_set, g_set):
    X_set = np.zeros((1, 2*len(q_set[0].feature)))
    y_set = []
    qg_index = np.zeros((1, 2))
    ran_labels = np.random.choice(np.unique(toLabelArray(q_set)), size=50, replace=False)
    for i, q_img in enumerate(q_set):
        if q_img.label in ran_labels:
            for j, g_img in enumerate(g_set):
                if j % 100 == 0:
                    print('Building MLP testing data... q:{:5d} g:{:5d}'.format(i, j))
                if g_img.label in ran_labels:
                    X_img = np.concatenate((q_img.feature, g_img.feature))
                    if g_img.label == q_img.label:
                        if g_img.camId != q_img.camId:
                            X_set = np.vstack((X_set, X_img))
                            y_set = y_set + [0]
                            qg_index = np.vstack((qg_index, np.asarray([i, j])))
                    else:
                        X_set = np.vstack((X_set, X_img))
                        y_set = y_set + [1]
                        qg_index = np.vstack((qg_index, np.asarray([i, j])))
    return X_set[1:,:], y_set, qg_index[1:,:]

def train_mlp(mlp, X_train, y_train, X_test, y_test, qg_index,
              q_set, g_set, k_nn_val):
    mlp.fit(X_train, y_train)
    y_learnt = mlp.predict_proba(X_test)
    qg_index = qg_index.astype(int)
    q_idx = [idx[0] for idx in qg_index]
    g_idx = [idx[1] for idx in qg_index]
    
    q_idx, q_g_idx = np.unique(q_idx, return_index=True)
    q_g_idx = np.append(q_g_idx, len(qg_index))
    g_q_idx = []
    g_q_dist = []
    for i in range(len(q_g_idx)-1):
        g_q_idx.append(g_idx[q_g_idx[i]: q_g_idx[i+1]])
        g_q_dist.append(y_learnt[:,1][q_g_idx[i]: q_g_idx[i+1]])
    
    nn_list = []
    for i, q in enumerate(q_idx):
        nn_list.append(np.asarray(g_q_idx[i])[np.argsort(g_q_dist[i])])
    
    q_mlp_set = list(np.asarray(q_set)[q_idx])
    knn_mlp_set = [list(np.asarray(g_set)[nn_idx]) for nn_idx, g_idx in zip(nn_list, g_q_idx)]
    for k in k_nn_val:
        knn_set = kNN(knn_mlp_set, 1)
        success_array = successArray(q_mlp_set, knn_set)
        success_rate = np.count_nonzero(success_array) / len(q_mlp_set)
        print ('[-Main] With {:2d}-NN, success rate is [{:.2%}]'.format(k, success_rate))
    mAP = mAPNN(q_mlp_set, knn_mlp_set)
    print('[--MLP] mAP is [{:.2%}]'.format(mAP))
    return mAP