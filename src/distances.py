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