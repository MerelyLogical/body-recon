# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:38:26 2018

@author: zw4215
"""

import scipy.io as sio
import json
import numpy as np

class DataProc:
    'data pre-processing'
    PATH = '../data/'
    
    @staticmethod
    def dataLoad():
        'loads all data in json as np array'
        JDATA = 'feature_data.json'
        with open(DataProc.PATH + JDATA, 'r') as jdata:
            features = json.load(jdata)
        return np.asarray(features)
    
    def labelLoad():
        'loads a all labels'
        LABEL = 'cuhk03_new_protocol_config_labeled.mat'
        lbl = sio.loadmat(DataProc.PATH + LABEL)
        idx_list = ['camId', 'filelist', 'gallery_idx'
                   ,'labels', 'query_idx', 'train_idx']
        return list(map(lambda x: lbl[x].flatten(), idx_list))

#data = DataProc.dataLoad()
query_idx = DataProc.labelLoad()