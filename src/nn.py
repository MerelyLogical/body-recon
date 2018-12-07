# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Tue Dec  4 18:28:15 2018

@author: MerelyLogical
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Nearest Neighbour
# ------------------------------------------------------------------------------
def neighbour(q_img, g_set, M, k, f_dist):
    'finds indexes of the k nearest neighbours in gallery'
    g_filtered =\
        [x for x in g_set if\
             x.label != q_img.label or x.camId != q_img.camId]
    d = [f_dist(q_img.feature, g_img.feature, M) for g_img in g_filtered]
    return [g_filtered[i] for i in np.argsort(d).tolist()[:k]]
    
def neighbours(q_set, g_set, M, k, f_dist):
    return [neighbour(q_img, g_set, M, k, f_dist) for q_img in q_set]

# ------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------
def successArray(q_set, k_set):
    'return the number of matches in k for every q'
    def sameLabelCount(img_a, img_b):
        if img_a.label == img_b.label:
            return 1
        else:
            return 0
        
    def perQueryRate(q_img, k_images):
        return sum([sameLabelCount(k_img, q_img) for k_img in k_images])
            
    return [perQueryRate(q_img, k_set[i]) for i, q_img in enumerate(q_set)]

def displayResults(query, k_result, k):
    'prints results for one query onto a figure'
    plt.figure()
    plt.subplot(1, k+1, 1)
    query.subplot()
    for i in range(k):
        plt.subplot(1, k+1, i+2)
        k_result[i].subplot()
    return None