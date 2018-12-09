# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:46:32 2018

@author: zw4215
"""

import numpy as np
from dataproc import toLabelArray
from train import train, train_rca

def build_tv(t_set, V_SIZE):
    v_labels = np.random.choice(toLabelArray(t_set), size=V_SIZE, replace=False)
    v_set = []
    nt_set = []
    for t_img in t_set:
        if t_img.label in v_labels:
            v_set.append(t_img)
        else:
            nt_set.append(t_img)
    vq_dict = {lbl:0 for lbl in v_labels}
    # REVISIT: how to pick vq (and thus vg) from v?
    return nt_set, vq_set, vg_set

def validate(t_set, k, rca_chunk, method):
    for i in range(k):
        nt_set, vq_set, vg_set = build_tv(t_set)
        if method == 'rca':
            train_rca(method, rca_chunk, nt_set, vq_set, vg_set)
        elif method in ['lmnn', 'mmc', 'mlkr']:
            train(method, nt_set, vq_set, vg_set)
        else:
            pass
