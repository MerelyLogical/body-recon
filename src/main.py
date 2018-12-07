# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Tue Dec  4 18:22:33 2018

@author: zw4215
"""

import numpy as np
import matplotlib.pyplot as plt
import metric_learn
import sklearn.decomposition as decomp
import sklearn.manifold as mnf

from dataproc  import dataLoad, splitData, toFeatureArray, toLabelArray, toImageArray
from distances import euclidean, mahalanobis
from nn        import neighbours, successArray, displayResults
from kmean     import kmean, linAssign, reassign
from perf      import start, lap

# ------------------------------------------------------------------------------
K = 10
M_PCA = 230
m_lmnn = np.zeros((M_PCA, M_PCA))
tr = start()

lap('Initialise', tr)
# ------------------------------------------------------------------------------

data, meta, idx = dataLoad()
t_set, q_set, g_set = splitData(data, meta, idx)
del data, meta, idx

#t_set = t_set
#q_set = q_set
#g_set = g_set

lap('Load data', tr)
# ------------------------------------------------------------------------------

pca = decomp.PCA(n_components=M_PCA)
t_set_pca_feature = pca.fit_transform(toFeatureArray(t_set))
#ratio = pca.explained_variance_ratio_
q_set_pca_feature = pca.transform(toFeatureArray(q_set))
g_set_pca_feature = pca.transform(toFeatureArray(g_set))
for i, t_img in enumerate(t_set):
    t_img.feature = t_set_pca_feature[i]
for i, q_img in enumerate(q_set):
    q_img.feature = q_set_pca_feature[i]
for i, g_img in enumerate(g_set):
    g_img.feature = g_set_pca_feature[i]
del t_set_pca_feature, q_set_pca_feature, g_set_pca_feature

lap('PCA', tr)
# -----------------------------------------------------------------------------

#lmnn = metric_learn.LMNN(k=3, min_iter=1, max_iter=10, learn_rate=1e-6, convergence_tol=1e-3, use_pca=False, verbose=True)
#t_set_lmnn_feature = lmnn.fit_transform(toFeatureArray(t_set), toLabelArray(t_set))
#q_set_lmnn_feature = lmnn.transform(toFeatureArray(q_set))
#g_set_lmnn_feature = lmnn.transform(toFeatureArray(g_set))
#for i, t_img in enumerate(t_set):
#    t_img.feature = t_set_lmnn_feature[i]
#for i, q_img in enumerate(q_set):
#    q_img.feature = q_set_lmnn_feature[i]
#for i, g_img in enumerate(g_set):
#    g_img.feature = g_set_lmnn_feature[i]
#del t_set_lmnn_feature, q_set_lmnn_feature, g_set_lmnn_feature
#
#lap('Train with LMNN', tr)
## ------------------------------------------------------------------------------
#
#mmc = metric_learn.mmc.MMC_Supervised(max_iter=10, convergence_threshold=1e-04, num_labeled=np.inf, num_constraints=100, verbose=True)
#t_set_mmc_feature = mmc.fit_transform(toFeatureArray(t_set), toLabelArray(t_set))
#q_set_mmc_feature = mmc.transform(toFeatureArray(q_set))
#g_set_mmc_feature = mmc.transform(toFeatureArray(g_set))
#for i, t_img in enumerate(t_set):
#    t_img.feature = t_set_mmc_feature[i]
#for i, q_img in enumerate(q_set):
#    q_img.feature = q_set_mmc_feature[i]
#for i, g_img in enumerate(g_set):
#    g_img.feature = g_set_mmc_feature[i]
#del t_set_mmc_feature, q_set_mmc_feature, g_set_mmc_feature
#
#lap('Train with MMC', tr)
# ------------------------------------------------------------------------------
#
#rca = metric_learn.rca.RCA(num_dims=None, pca_comps=None)
#chuncky = np.repeat(list(range(1, 25)), 307)
#rca.fit(toFeatureArray(t_set), chuncky)
#t_set_rca_feature = rca.transform(toFeatureArray(t_set))
#q_set_rca_feature = rca.transform(toFeatureArray(q_set))
#g_set_rca_feature = rca.transform(toFeatureArray(g_set))
#for i, t_img in enumerate(t_set):
#    t_img.feature = t_set_rca_feature[i]
#for i, q_img in enumerate(q_set):
#    q_img.feature = q_set_rca_feature[i]
#for i, g_img in enumerate(g_set):
#    g_img.feature = g_set_rca_feature[i]
#del t_set_rca_feature, q_set_rca_feature, g_set_rca_feature
#
#lap('Train with RCA', tr)
# ------------------------------------------------------------------------------
#
#mlkr = metric_learn.mlkr.MLKR(num_dims=200, A0=None, tol=1e-6, max_iter=10, verbose=True)
#t_set_mlkr_feature = mlkr.fit_transform(toFeatureArray(t_set), toLabelArray(t_set))
#q_set_mlkr_feature = mlkr.transform(toFeatureArray(q_set))
#g_set_mlkr_feature = mlkr.transform(toFeatureArray(g_set))
#for i, t_img in enumerate(t_set):
#    t_img.feature = t_set_mlkr_feature[i]
#for i, q_img in enumerate(q_set):
#    q_img.feature = q_set_mlkr_feature[i]
#for i, g_img in enumerate(g_set):
#    g_img.feature = g_set_mlkr_feature[i]
#del t_set_mlkr_feature, q_set_mlkr_feature, g_set_mlkr_feature
#
#lap('Train with MLKR', tr)
# ------------------------------------------------------------------------------
# This one is shit
#
#mds = mnf.MDS(n_components=3, metric=True, n_init=3, max_iter=30, verbose=2, eps=0.001, n_jobs=None, random_state=None, dissimilarity='euclidean')
#q_set_mds_feature = mds.fit_transform(toFeatureArray(q_set))
#g_set_mds_feature = mds.fit_transform(toFeatureArray(g_set))
#for i, q_img in enumerate(q_set):
#    q_img.feature = q_set_mds_feature[i]
#for i, g_img in enumerate(g_set):
#    g_img.feature = g_set_mds_feature[i]
#del q_set_mds_feature, g_set_mds_feature
#
#lap('Train with MDS', tr)
# ------------------------------------------------------------------------------

knn_set = neighbours(q_set, g_set, m_lmnn, K, euclidean)

lap('Calculate {}-NN'.format(K), tr)
# ------------------------------------------------------------------------------

success_array = successArray(q_set, knn_set)
success_rate = np.count_nonzero(success_array) / len(q_set)
print ('[*Main] With {}-NN, success rate is {}'.format(K, success_rate))

lap('Evaluate NN success rate', tr)
# ------------------------------------------------------------------------------

#for i in range(3):
#    displayResults(q_set[i], knn_set[i], K)
#
#lap('Print NN results', tr)
# ------------------------------------------------------------------------------

#km_set, km_g_labels = kmean(g_set)
#ass_mtx = linAssign(km_g_labels, g_set)
#km_reassigned_set = reassign(km_set, ass_mtx)
#kmeans_set = neighbours(q_set, km_reassigned_set, K, euclidean)
#success_array = successArray(q_set, kmeans_set)
#success_rate = np.count_nonzero(success_array) / len(q_set)
#print ('[*Main] With {}-means, success rate is {}'.format(K, success_rate))
#
#lap('Calculate k-means', tr)




