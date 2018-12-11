# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:46:32 2018

@author: zw4215
"""

import copy
import numpy as np
from dataproc import toLabelArray
from train import train, train_rca

def build_qg(v_set):
    'build query and gallery from validation set'
    seen = []
    vq_set = []
    vg_set = []
    for v_img in v_set:
        if (v_img.label, v_img.camId) in seen:
            vg_set.append(v_img)
        else:
            vq_set.append(v_img)
            seen.append((v_img.label, v_img.camId))
    return vq_set, vg_set

def build_tv(t_set, V_SIZE):
    'build validation set from training set'
    v_labels = np.random.choice(toLabelArray(t_set), size=V_SIZE, replace=False)
    v_set = []
    nt_set = []
    for t_img in t_set:
        if t_img.label in v_labels:
            v_set.append(copy.deepcopy(t_img))
        else:
            nt_set.append(copy.deepcopy(t_img))
    tracker = {}
    # get rid of those who only appeared in one camera
    for v_img in v_set:
        if v_img.label in tracker:
            if v_img.camId == tracker[v_img.label]:
                tracker[v_img.label] = -1
        else:
            tracker[v_img.label] = v_img.camId
    nv_set = [v_img for v_img in v_set if tracker[v_img.label] == -1]
    vq_set, vg_set = build_qg(nv_set)
    return nt_set, vq_set, vg_set

#def validate(t_set, k, rca_chunk, method):
#    'validate'
#    for i in range(k):
#        nt_set, vq_set, vg_set = build_tv(t_set)
#        if method == 'rca':
#            train_rca(method, rca_chunk, nt_set, vq_set, vg_set)
#        elif method in ['lmnn', 'mmc', 'mlkr']:
#            train(method, nt_set, vq_set, vg_set)
#        else:
#            pass
