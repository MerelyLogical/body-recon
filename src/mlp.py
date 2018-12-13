# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:52:38 2018

@author: zw4215
"""

import numpy as np
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
            print('Building MLP data... {:5d}'.format(i))
        X_img, y_img = build_sd_sets(t_img, t_set)
        X_set = np.vstack((X_set, X_img))
        y_set = y_set + y_img
    return X_set[1:,:], y_set

def build_mlp_test(q_set, g_set):
    X_set = np.zeros((1, 2*len(q_set[0].feature)))
    y_set = []
    qg_index = np.zeros((1, 2))
    ran_labels = np.random.choice(np.unique(toLabelArray(q_set)), size=10, replace=False)
    for i, q_img in enumerate(q_set):
        if q_img.label in ran_labels:
            for j, g_img in enumerate(g_set):
                if j % 100 == 0:
                    print('Building MLP data... {:5d}{:5d}'.format(i, j))
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
#
#X = np.load('npy/mlp_X.npy')
#y = np.load('npy/mlp_y.npy')
#X_test = np.load('npy/mlp_X_test.npy')
#y_test = np.load('npy/mlp_y_test.npy')
#
#data, meta, idx = dataLoad()
#t_set, q_set, g_set = splitData(data, meta, idx)
#del data, meta, idx


#mlp = MLPClassifier(hidden_layer_sizes=(20,10,5,), activation='relu',
#                    solver='adam', alpha=0.0001, batch_size='auto',
#                    learning_rate='constant', learning_rate_init=0.001,
#                    power_t=0.5, max_iter=200, shuffle=True, random_state=None,
#                    tol=0.0001, verbose=True, warm_start=False, momentum=0.9,
#                    nesterovs_momentum=True, early_stopping=False,
#                    validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
#                    epsilon=1e-08)
#
#mlp.fit(X, y)
#y_learnt = mlp.predict_proba(X_test)
