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
    k = max(k_labels) + 1
    # the label is used in assignment. its content is the original label
    g_label_translator = np.unique(g_labels)
    cost_mtx = np.zeros((k, k))
    for k_lbl, g_lbl in zip(k_labels, g_labels):
        cost_mtx[np.where(g_label_translator == g_lbl)[0][0]][k_lbl] -= 1
    assign = linear_assignment(cost_mtx)
    ass_mtx = np.zeros(k)
    for a in assign:
        ass_mtx[a[1]] = g_label_translator[a[0]]
    return ass_mtx

def reassign(km_set, ass_mtx):
    return [Image(km_img.feature, -1, '', a_lbl)\
            for km_img, a_lbl in zip(km_set, ass_mtx)]
