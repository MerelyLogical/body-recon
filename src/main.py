# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Tue Dec  4 18:22:33 2018

@author: zw4215
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as decomp

from dataproc  import dataLoad, splitData, toFeatureArray, toLabelArray, toImageArray
from distances import euclidean, trainLMNN, mahalanobis
from nn        import neighbours, successArray, displayResults
from kmean     import kmean, linAssign, reassign
from perf      import start, lap

# ------------------------------------------------------------------------------
K = 10
M_PCA = 400
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
ratio = pca.explained_variance_ratio_
plt.plot(ratio)
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
# ------------------------------------------------------------------------------

m_lmnn = trainLMNN(t_set)

lap('Train with LMNN', tr)
# ------------------------------------------------------------------------------

knn_set = neighbours(q_set, g_set, m_lmnn, K, mahalanobis)

lap('Calculate {}-NN'.format(K), tr)
# ------------------------------------------------------------------------------

success_array = successArray(q_set, knn_set)
success_rate = np.count_nonzero(success_array) / len(q_set)
print ('[*Main] With {}-NN, success rate is {}'.format(K, success_rate))

lap('Evaluate NN success rate', tr)
# ------------------------------------------------------------------------------

for i in range(3):
    displayResults(q_set[i], knn_set[i], K)

lap('Print NN results', tr)
# ------------------------------------------------------------------------------

#km_set, km_g_labels = kmean(g_set)
#ass_mtx = linAssign(km_g_labels, g_set)
#km_reassigned_set = reassign(km_set, ass_mtx)
#kmeans_set = neighbours(q_set, km_reassigned_set, K, euclidean)
#success_array = successArray(q_set, kmeans_set)
#success_rate = np.count_nonzero(success_array) / len(q_set)
#print ('[*Main] With {}-means, success rate is {}'.format(K, success_rate))

lap('Calculate k-means', tr)




