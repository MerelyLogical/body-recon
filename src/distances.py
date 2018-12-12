# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Tue Dec  4 18:33:02 2018

@author: MerelyLogical
"""

import numpy as np

# ------------------------------------------------------------------------------
# Distances
# ------------------------------------------------------------------------------
def euclidean(x, y):
    return np.linalg.norm(y-x)

def chessboard(x, y):
    return np.max(np.absolute(y-x))

def manhattan(x, y):
    return np.sum(np.absolute(y-x))

def cosine(x, y):
    return np.dot(y, x)/(np.linalg.norm(y)*np.linalg.norm(x))
    
#def mahalanobis(x, y, M):
#    d = x-y
#    return np.dot(np.dot(d.transpose(), M), d)