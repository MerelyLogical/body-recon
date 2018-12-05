# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Wed Dec  5 14:22:40 2018

@author: zw4215
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
from dataproc import Image, toFeatureArray, toLabelArray, toImageArray

def kmean(g_set):
    'returns centers after k-means'
    k = len(np.unique(toLabelArray(g_set)))
    cl = KMeans(n_clusters = k)
    cl.fit(toFeatureArray(g_set))
    return toImageArray(cl.cluster_centers_, list(range(1, 701))), cl.labels_

def linAssign(k_labels, g_set):
    'returns reassigned kmean_label'
    g_labels = toLabelArray(g_set)
    # this is done to match array index with label. index 0 is thus unused
    j = max(g_labels) + 1
    k = max(k_labels) + 1
    cost_mtx = np.zeros((j, k))
    for k_lbl, g_lbl in zip(k_labels, g_labels):
        cost_mtx[g_lbl][k_lbl] -= 1
    assign = linear_assignment(cost_mtx)
    ass_mtx = np.zeros(k)
    for a in assign:
        ass_mtx[a[1]]= a[0]
    return ass_mtx

def reassign(km_set, ass_mtx):
    return [Image(km_img.feature, -1, '', a_lbl)\
            for km_img, a_lbl in zip(km_set, ass_mtx)]
