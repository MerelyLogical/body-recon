# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Tue Dec  4 18:22:33 2018

@author: zw4215
"""

import numpy as np
import metric_learn
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.manifold import TSNE

from dataproc  import dataLoad, splitData
from distances import euclidean
from nn        import allNN, kNN, mAPNN, successArray, displayResults
from kmean     import kmean, linAssign, reassign
from perf      import start, lap
from train     import train, train_rca, unsup_transform

# ------------------------------------------------------------------------------
# Setting parameters
print('[--Sys]-----------------------------------------------------------START')

K_NN = int(input('K-NN [1]: ') or '1')
K_MEANS = int(input('K-means [1]: ') or '1')
s = input('Use PCA? (please say yes) [Y]/N: ') or 'Y'
if s.lower().strip() in ['n', 'no', '0']:
    use_pca = False
    M_PCA = 1
else:
    use_pca = True
    M_PCA = int(input('M_PCA [230]:') or '230')
s = input('Use kernel? Y/[N]: ') or 'N'
if s.lower().strip() in ['y', 'yes', '1']:
    use_kernel = True
else:
    use_kernel = False
s = input('Use t-SNE? Y/[N]: ') or 'N'
if s.lower().strip() in ['y', 'yes', '1']:
    use_tsne = True
else:
    use_tsne = False
train_method = input('training method [none]/lmnn/mmc/rca/mlkr:') or 'none'
# ------------------------------------------------------------------------------
# Initialise
print('[--Sys]---------------------------------------------------------LOADING')

tr = start()
# Trainers
pca = PCA(n_components=M_PCA)
kernel = RBFSampler(gamma=1.0, n_components=230, random_state=None)
tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=250, n_iter_without_progress=100, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=2, random_state=None, method='barnes_hut', angle=0.5)
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

if use_kernel:
    train(kernel, t_set, q_set, g_set)
    lap('Kernel', tr)

if use_tsne:
    unsup_transform(tsne, t_set, q_set, g_set)
    lap('TSNE', tr)

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

print('[--Sys]-------------------------------------------------------------END')