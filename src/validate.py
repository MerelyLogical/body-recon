# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:46:32 2018

@author: zw4215
"""

import copy
import numpy as np
from dataproc import toLabelArray

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
        if (v_img.label, v_img.camId) in tracker:
            if tracker[(v_img.label, v_img.camId)] == 1:
                tracker[(v_img.label, v_img.camId)] = 2
        else:
            tracker[(v_img.label, v_img.camId)] = 1
    nv_set = []
    for v_img in v_set:
        try:
            b = tracker[(v_img.label, 1)] == 2 and tracker[(v_img.label, 2)] == 2
        except KeyError:
            b = False
        if b:
            nv_set.append(v_img)
    vq_set, vg_set = build_qg(nv_set)
    return nt_set, vq_set, vg_set
