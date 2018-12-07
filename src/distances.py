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
def euclidean(x, y, M):
    return np.linalg.norm(y-x)

def mahalanobis(x, y, M):
    d = x-y
    return np.dot(np.dot(d.transpose(), M), d)