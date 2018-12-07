# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Tue Dec  4 18:33:02 2018

@author: MerelyLogical
"""

import numpy as np
import metric_learn

from dataproc import toFeatureArray, toLabelArray

# ------------------------------------------------------------------------------
# Distances
# ------------------------------------------------------------------------------
def euclidean(x, y, M):
    return np.linalg.norm(y-x)

def trainLMNN(t_set):
    lmnn = metric_learn.LMNN(k=3, max_iter=100, learn_rate=1e-5, convergence_tol=1e-2, use_pca=False)
    lmnn.fit(toFeatureArray(t_set), toLabelArray(t_set))
    return lmnn.metric()

def mahalanobis(x, y, M):
    d = x-y
    return np.dot(np.dot(d.transpose(), M), d)