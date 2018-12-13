# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Thu Dec 13 00:06:10 2018

@author: zw4215
"""

import numpy as np
from nn import displayResults

# ------------------------------------------------------------------------------
# Initialise
N_PIC = 3
K_NN = 10

# ------------------------------------------------------------------------------
# Load data
base_cfg = '1000_none_euclidean'
test_cfg = '1000_lmnn_euclidean'
base_q_set = np.load('npy/q_{}.npy'.format(base_cfg))
base_knn_set = np.load('npy/d_{}.npy'.format(base_cfg))
test_q_set = np.load('npy/q_{}.npy'.format(test_cfg))
test_knn_set = np.load('npy/d_{}.npy'.format(test_cfg))

# ------------------------------------------------------------------------------
# Print K-NN pictures

pic_idx = np.random.choice(len(base_q_set), size=N_PIC, replace=False)
for i in pic_idx:
    displayResults(base_q_set[i], test_knn_set[i], K_NN)
