# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Tue Dec  4 18:22:33 2018

@author: zw4215
"""

#Todo
#tranformation picture
#validate
#memory

import numpy as np
import metric_learn
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.manifold import TSNE

from dataproc  import dataLoad, splitData
from distances import euclidean, chessboard, manhattan, cosine
from nn        import allNN, kNN, mAPNN, successArray
from kmean     import kmean, linAssign, reassign
from perf      import start, lap
from train     import train, train_rca, unsup_transform
from validate  import build_tv

# ------------------------------------------------------------------------------
# Setting parameters
print('[--Sys]-----------------------------------------------------------START')

def defaultNo(s):
    if s.lower().strip() in ['y', 'yes', '1']:
        return True
    else:
        return False

def defaultYes(s):
    if s.lower().strip() in ['n', 'no', '0']:
        return False
    else:
        return True

# values of k-nn
k_nn_val = [1, 3, 5, 7, 10]

s = input('Use PCA? (please say yes) [Y]/N: ') or 'Y'
use_pca = defaultYes(s)
if use_pca:
    M_PCA = int(input('M_PCA [230]: ') or '230')
else:
    M_PCA = -1

s = input('Use 5-fold validation for M_PCA? Y/[N]: ') or 'N'
use_val = defaultNo(s)
# values for M_pca in 5 fold validation
m_val = [190, 210, 230, 250, 270]

s = input('Use kernel? Y/[N]: ') or 'N'
use_kernel = defaultNo(s)

s = input('Use t-SNE? Y/[N]: ') or 'N'
use_tsne = defaultNo(s)

train_method = input('training method [none]/lmnn/mmc/rca/mlkr/itml: ') or 'none'

s = input('Use K-means in addition to K-NN? Y/[N]: ') or 'N'
use_kmeans = defaultNo(s)

distance_method = input('distance method [euclidean]/chessboard/manhattan/cosine: ') or 'euclidean'
if distance_method == 'euclidean':
    f_dist = euclidean
elif distance_method == 'chessboard':
    f_distance = chessboard
elif distance_method == 'manhattan':
    f_dist = manhattan
elif distance_method == 'cosine':
    f_dist = cosine
else:
    f_dist = euclidean
# REVISIT: distance of tsne can be changed too.
# but euclidean still semms to perform the best

N_PIC = int(input('How many result pictures to print? [0]: ') or '0')

# ------------------------------------------------------------------------------
# Initialise

tr = start()

# Trainers
pca = PCA(n_components=M_PCA)

kernel = RBFSampler(gamma=1.0, n_components=230, random_state=None)

tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,
            learning_rate=200.0, n_iter=250, n_iter_without_progress=100,
            min_grad_norm=1e-07, metric='euclidean', init='random', verbose=2,
            random_state=None, method='barnes_hut', angle=0.5)

lmnn = metric_learn.LMNN(k=3, min_iter=1, max_iter=10, learn_rate=1e-6,
                         convergence_tol=1e-3, use_pca=False, verbose=True)

mmc = metric_learn.mmc.MMC_Supervised(max_iter=10, convergence_threshold=1e-04,
                                      num_labeled=np.inf, num_constraints=100,
                                      verbose=True)

rca = metric_learn.rca.RCA(num_dims=None, pca_comps=None)
chuncky = np.repeat(list(range(1, 25)), 307)

mlkr = metric_learn.mlkr.MLKR(num_dims=200, A0=None, tol=1e-6, max_iter=10,
                              verbose=True)

itml = metric_learn.itml.ITML_Supervised(gamma=1.0, max_iter=10,
                                         convergence_threshold=0.001,
                                         num_labeled=np.inf,
                                         num_constraints=100, bounds=None,
                                         A0=None, verbose=True)

# ------------------------------------------------------------------------------
# Load data
print('[--Sys]---------------------------------------------------------LOADING')

data, meta, idx = dataLoad()
t_set, q_set, g_set = splitData(data, meta, idx)
del data, meta, idx

lap('Load data', tr)
# ------------------------------------------------------------------------------
# Validation

print('[-Vldt]------------------------------------------------------VALIDATION')

if use_val:
    result_val = []
    for i in range(5):
        nt_set, vq_set, vg_set = build_tv(t_set, 100)
        pca = PCA(n_components=m_val[i])
        train(pca, nt_set, vq_set, vg_set)
        lap('Perform PCA', tr)
        nn_vg_set = allNN(vq_set, vg_set, f_dist)
        lap('Calculate all pair-wise distances for NN', tr)
        vmAP = mAPNN(vq_set, nn_vg_set)
        print('[-Main] mAP is [{:.2%}]'.format(vmAP))
        lap('Calculate mAP with NN', tr)
        result_val.append(vmAP)
        lap('Validating PCA, iter: {}'.format(i), tr)
    
    M_PCA = m_val[np.argmax(vmAP)]
    pca = PCA(n_components=M_PCA)

else:
    print('[-Vldt] Skip PCA validation')
lap('Validation', tr)
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

elif train_method == 'itml':
    train(itml, t_set, q_set, g_set)
    lap('Train with ITML', tr)
    
else:
    lap('Skip training', tr)

# ------------------------------------------------------------------------------
# NN
print('[---NN]------------------------------------------------------K-NN & mAP')

nn_g_set = allNN(q_set, g_set, f_dist)
lap('Calculate all pair-wise distances for NN', tr)
# ------------------------------------------------------------------------------
# K-NN

for k in k_nn_val:
    knn_set = kNN(nn_g_set, k)
    success_array = successArray(q_set, knn_set)
    success_rate = np.count_nonzero(success_array) / len(q_set)
    print ('[-Main] With {:2d}-NN, success rate is [{:.2%}]'.format(k, success_rate))

# ------------------------------------------------------------------------------
# mAP

mAP = mAPNN(q_set, nn_g_set)
print('[-Main] mAP is [{:.2%}]'.format(mAP))

lap('Calculate mAP with NN', tr)
# ------------------------------------------------------------------------------
# K-means
print('[kmean]---------------------------------------------------------K-MEANS')

if use_kmeans:
    km_set, km_g_labels = kmean(g_set)
    ass_mtx = linAssign(km_g_labels, g_set)
    km_reassigned_set = reassign(km_set, ass_mtx)
    kmean_g_set = allNN(q_set, km_reassigned_set, f_dist)
    for k in k_nn_val:
        kmeans_set = kNN(kmean_g_set, k)
        success_array = successArray(q_set, kmeans_set)
        success_rate = np.count_nonzero(success_array) / len(q_set)
        print ('[*Main] With {:2d}-means, success rate is [{:.2%}]'.format(k, success_rate))
    
    lap('Calculate k-means', tr)
# ------------------------------------------------------------------------------
# mAP
    
    mAP = mAPNN(q_set, kmean_g_set)
    print('[-Main] mAP is [{:.2%}]'.format(mAP))
    
    lap('Calculate mAP with NN', tr)
else:
    print('[kmean] Skip K-means')

# ------------------------------------------------------------------------------
# Saving data
print('[-Save]--------------------------------------------------SAVING RESULTS')
cfgstr = '_{}{}{}{}_{}_{}'.format(int(use_pca),int(use_val),int(use_kernel),
          int(use_tsne),train_method,distance_method)
np.save('npy/q'+cfgstr, q_set)
np.save('npy/d'+cfgstr, knn_set)

print('[--Sys]-------------------------------------------------------------END')