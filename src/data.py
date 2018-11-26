# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:38:26 2018

@author: zw4215
"""

import scipy.io as sio
import json
import numpy as np

# data pre-processing
class DataProc:
    PATH = '../data/'
    
    @staticmethod
    # loads all data in json as np array
    def dataLoad():
        JDATA = 'feature_data.json'
        with open(DataProc.PATH + JDATA, 'r') as jdata:
            features = json.load(jdata)
        return np.asarray(features)
    
    # loads a single label
    def labelLoad(idx_name):
        LABEL = 'cuhk03_new_protocol_config_labeled.mat'
        lbl = sio.loadmat(DataProc.PATH + LABEL)
        print(idx_name)
        return lbl[idx_name].flatten()
        
    # loads all labels
    def all_label():
        idx_list = ['train_idx', 'query_idx']
        return list(map(DataProc.labelLoad, idx_list))

#data = DataProc.dataLoad()
train_idx = DataProc.all_label()