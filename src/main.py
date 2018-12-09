# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Tue Dec  4 18:22:33 2018

@author: zw4215
"""

import numpy as np
import metric_learn
import sklearn.decomposition as decomp

from dataproc  import dataLoad, splitData
from distances import euclidean
from nn        import allNN, kNN, mAPNN, successArray, displayResults
from kmean     import kmean, linAssign, reassign
from perf      import start, lap
from train     import train, train_rca

# ------------------------------------------------------------------------------
print('[--Sys]-----------------------------------------------------------START')
# Setting parameters

K_NN = int(input('K-NN [5]: ') or '5')
K_MEANS = int(input('K-means [5]: ') or '5')
s = input('Use-pca [Y]/N: ') or 'Y'
if s.lower().strip() in ['n', 'no', '0']:
    use_pca = False
    M_PCA = 1
else:
    use_pca = True
    M_PCA = int(input('M_PCA [230]:') or '230')
train_method = input('training method [none]/lmnn/mmc/rca/mlkr:') or 'none'
# ------------------------------------------------------------------------------
# Initialise
print('[--Sys]-----------------------------------------------------------START')
tr = start()
# Trainers
pca = decomp.PCA(n_components=M_PCA)
lmnn = metric_learn.LMNN(k=3, min_iter=1, max_iter=10, learn_rate=1e-6, convergence_tol=1e-3, use_pca=False, verbose=True)
mmc = metric_learn.mmc.MMC_Supervised(max_iter=10, convergence_threshold=1e-04, num_labeled=np.inf, num_constraints=100, verbose=True)
rca = metric_learn.rca.RCA(num_dims=None, pca_comps=None)
chuncky = np.repeat(list(range(1, 25)), 307)
mlkr = metric_learn.mlkr.MLKR(num_dims=200, A0=None, tol=1e-6, max_iter=10, verbose=True)

lap('Initialise', tr)
# ------------------------------------------------------------------------------
# Load data

data, meta, idx = dataLoad()
t_set, q_set, g_set = splitData(data, meta, idx)
del data, meta, idx

lap('Load data', tr)
# ------------------------------------------------------------------------------
# Training
print('[Train]--------------------------------------------------------TRAINING')

if use_pca:
    train(pca, t_set, q_set, g_set)
    lap('PCA', tr)

if train_method == 'lmnn':
    train(lmnn, t_set, q_set, g_set)
    lap('Train with LMNN', tr)
    
elif train_method == 'mmc':
    train(mmc, t_set, q_set, g_set)
    lap('Train with MMC', tr)
    
elif train_method == 'rca':
    train_rca(rca, chuncky, t_set, q_set, g_set)
    lap('Train with RCA', tr)
    
elif train_method == 'mlkr':
    train(mlkr, t_set, q_set, g_set)
    lap('Train with MLKR', tr)
    
else:
    lap('Skip training', tr)

# ------------------------------------------------------------------------------
# NN
print('[---NN]------------------------------------------------------K-NN & mAP')

nn_g_set = allNN(q_set, g_set, euclidean)
lap('Calculate all pair-wise distances for NN'.format(K_NN), tr)
# ------------------------------------------------------------------------------
# K-NN

knn_set = kNN(nn_g_set, K_NN)
success_array = successArray(q_set, knn_set)
success_rate = np.count_nonzero(success_array) / len(q_set)
print ('[-Main] With {}-NN, success rate is [{:.2%}]'.format(K_NN, success_rate))

for i in range(3):
    displayResults(q_set[i], knn_set[i], K_NN)

lap('Evaluate NN success rate', tr)
# ------------------------------------------------------------------------------
# mAP

mAP = mAPNN(q_set, nn_g_set)
print('[-Main] mAP is [{:.2%}]'.format(mAP))

lap('Calculate mAP with NN', tr)
# ------------------------------------------------------------------------------
# K-means
print('[kmean]---------------------------------------------------------K-MEANS')

km_set, km_g_labels = kmean(g_set)
ass_mtx = linAssign(km_g_labels, g_set)
km_reassigned_set = reassign(km_set, ass_mtx)
kmean_g_set = allNN(q_set, km_reassigned_set, euclidean)
kmeans_set = kNN(kmean_g_set, K_MEANS)
success_array = successArray(q_set, kmeans_set)
success_rate = np.count_nonzero(success_array) / len(q_set)
print ('[*Main] With {}-means, success rate is [{:.2%}]'.format(K_MEANS, success_rate))

lap('Calculate k-means', tr)
print('[--Sys]-------------------------------------------------------------END')