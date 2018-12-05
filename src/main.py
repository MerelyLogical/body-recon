# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Tue Dec  4 18:22:33 2018

@author: zw4215
"""

import numpy as np

from dataproc  import dataLoad, splitData
from distances import euclidean
from nn        import neighbours, successArray, displayResults
from kmean     import kmean, linAssign, reassign
from perf      import start, lap

# ------------------------------------------------------------------------------
K = 10
tr = start()

lap('Initialise', tr)
# ------------------------------------------------------------------------------

data, meta, idx = dataLoad()
t_set, q_set, g_set = splitData(data, meta, idx)
del data, meta, idx

lap('Load data', tr)
# ------------------------------------------------------------------------------

#knn_set = neighbours(q_set, g_set, K, euclidean)

lap('Calculate {}-NN'.format(K), tr)
# ------------------------------------------------------------------------------

#success_array = successArray(q_set, knn_set)
#success_rate = np.count_nonzero(success_array) / len(q_set)
#print ('[*Main] With {}-NN, success rate is {}'.format(K, success_rate))

lap('Evaluate NN success rate', tr)
# ------------------------------------------------------------------------------

#for i in range(3):
#    displayResults(q_set[i], knn_set[i], K)

lap('Print NN results', tr)
# ------------------------------------------------------------------------------

km_set, km_g_labels = kmean(g_set)
ass_mtx = linAssign(km_g_labels, g_set)
km_reassigned_set = reassign(km_set, ass_mtx)
kmeans_set = neighbours(q_set, km_reassigned_set, K, euclidean)
success_array = successArray(q_set, kmeans_set)
success_rate = np.count_nonzero(success_array) / len(q_set)
print ('[*Main] With {}-means, success rate is {}'.format(K, success_rate))

lap('Calculate k-means', tr)




