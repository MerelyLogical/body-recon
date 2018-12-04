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
from perf      import start, lap

# ------------------------------------------------------------------------------
K = 10
tr = start

lap('Initialise', tr)
# ------------------------------------------------------------------------------

data, meta, idx = dataLoad()
t_set, q_set, g_set = splitData(data, meta, idx)
del data, meta, idx

lap('Load data', tr)
# ------------------------------------------------------------------------------

k_set = neighbours(q_set, g_set, K, euclidean)

lap('Calculate 10-NN', tr)
# ------------------------------------------------------------------------------

success_array = successArray(q_set, k_set)
success_rate = np.count_nonzero(success_array) / len(q_set)
print ('[*Main] With {}-NN, success rate is {}'.format(K, success_rate))

lap('Evaluate success rate', tr)
# ------------------------------------------------------------------------------

for i in range(3):
    displayResults(q_set[i], k_set[i], K)

lap('Print results', tr)
# ------------------------------------------------------------------------------