# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Thu Dec 13 00:06:10 2018

@author: zw4215
"""

import numpy as np
import matplotlib.pyplot as plt

def displayResults(query, k_result, k):
    'prints results for one query onto a figure'
    plt.figure()
    plt.subplot(1, k+1, 1)
    query.subplot()
    for i in range(k):
        plt.subplot(1, k+1, i+2)
        k_result[i].subplot()
    return None

# ------------------------------------------------------------------------------
# Initialise

N_PIC = 2
K_NN = 10
base_cfg = '1000_none_euclidean'
test_cfg = '1000_lmnn_euclidean'

# ------------------------------------------------------------------------------
# Load data

base_q_set = np.load('npy/q_{}.npy'.format(base_cfg))
base_knn_set = np.load('npy/d_{}.npy'.format(base_cfg))
test_q_set = np.load('npy/q_{}.npy'.format(test_cfg))
test_knn_set = np.load('npy/d_{}.npy'.format(test_cfg))

# ------------------------------------------------------------------------------
# Print K-NN pictures

pic_idx = np.random.choice(len(base_q_set), size=N_PIC, replace=False)
for i in pic_idx:
    displayResults(base_q_set[i], base_knn_set[i], K_NN)
    displayResults(test_q_set[i], test_knn_set[i], K_NN)

