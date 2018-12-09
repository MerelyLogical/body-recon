# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Tue Dec  4 18:28:15 2018

@author: zw4215
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Nearest Neighbour
# ------------------------------------------------------------------------------
def neighbour(q_img, g_set, f_dist, i):
    'finds indexes of the k nearest neighbours in gallery'
    if(i % 300 == 0):
        print ('[---NN] Querying {}/1400'.format(i))
    g_filtered =\
        [x for x in g_set if\
             x.label != q_img.label or x.camId != q_img.camId]
    d = [f_dist(q_img.feature, g_img.feature) for g_img in g_filtered]
    return [g_filtered[i] for i in np.argsort(d).tolist()]
    
def allNN(q_set, g_set, f_dist):
    'list of gallery images sorted by distance for each query'
    return [neighbour(q_img, g_set, f_dist, i) for i, q_img in enumerate(q_set)]

def kNN(nn_g_set, k):
    'cuts off and only keep the first k image in the gallery'
    return [q_nn[:k] for q_nn in nn_g_set]

def averagePrecision(counter):
    'returns AP value from counter'
    p = counter[-1]
    recall = np.asarray([t/p for t in counter])
    precision = np.asarray([t/(i+1) for i,t in enumerate(counter)])
    r, r_idx = np.unique(recall, return_index=True)
    p = precision[r_idx]
    xs = np.linspace(0, 1, 11)
    interpolated_p = [np.interp(x, r, p) for x in xs]
    return np.average(interpolated_p)

def mAPNN(q_set, nn_g_set):
    'changes gallery into list of 1 and 0'
    q_g_nn_set = nn_g_set
    aps = []
    for q_idx, g_nn_set in enumerate(q_g_nn_set):
        counter = np.zeros(len(g_nn_set))
        for g_idx, g_img in enumerate(g_nn_set):
            if g_img.label == q_set[q_idx].label:
                counter[g_idx] += 1
        ap = averagePrecision(np.cumsum(counter))
        aps.append(ap)
    return np.average(aps)

# ------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------
def successArray(q_set, k_set):
    'return the number of matches in k for every q'
    def sameLabelCount(img_a, img_b):
        if img_a.label == img_b.label:
            return 1
        else:
            return 0
        
    def perQueryRate(q_img, k_images):
        return sum([sameLabelCount(k_img, q_img) for k_img in k_images])
            
    return [perQueryRate(q_img, k_set[i]) for i, q_img in enumerate(q_set)]

def displayResults(query, k_result, k):
    'prints results for one query onto a figure'
    plt.figure()
    plt.subplot(1, k+1, 1)
    query.subplot()
    for i in range(k):
        plt.subplot(1, k+1, i+2)
        k_result[i].subplot()
    return None