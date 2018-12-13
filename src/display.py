# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Thu Dec 13 00:06:10 2018

@author: zw4215
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Initialise

N_PIC = 10
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
    f, axarr = plt.subplots(2, K_NN+1)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.9, wspace=0.1, hspace=0.1)
    [ax.axis('off') for ax in axarr.ravel()]
    base_q_set[i].subplot(axarr[0,0])
    for j in range(K_NN):
        base_knn_set[i][j].subplot(axarr[0,j+1])
        test_knn_set[i][j].subplot(axarr[1,j+1])
