# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Mon Nov 26 16:38:26 2018

@author: zw4215
"""

import scipy.io as sio
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Image(object):
    def __init__ (self, feature, camId, path, label):
        self.feature = feature
        self.camId = camId
        self.path = path
        self.label = label
        
    def __str__ (self):
        PATH = '../data/images_cuhk03/'
        plt.imshow(mpimg.imread(PATH + self.path))
        display = 'Class: ' + str(self.label) + '\n'\
            + 'Camera: ' + str(self.camId) + '\n'\
            + 'Path: ' + str(self.path)
        return display
    
def dataLoad():
    'loads all data'
    
    PATH = '../data/'
    JDATA = 'feature_data.json'
    LABEL = 'cuhk03_new_protocol_config_labeled.mat'
    
    with open(PATH + JDATA, 'r') as jdata:
        features = json.load(jdata)
    data = np.asarray(features)
    lbl = sio.loadmat(PATH + LABEL)
    
    meta_list = ['camId', 'filelist', 'labels']
    idx_list = ['train_idx', 'query_idx', 'gallery_idx']
    
    def loadMeta(x):
        return lbl[x].flatten()
    def loadIdx(x):
        return lbl[x].flatten()-1
    
    meta = [loadMeta(x) for x in meta_list]
    idx = [loadIdx(x) for x in idx_list]
    
    return data, meta, idx

def splitData(data, meta, idx):
    'splits data into train, query, gallery'
    
    def buildImage(x):
        return Image(data[x], meta[0][x], meta[1][x][0], meta[2][x])
    def buildImageList(x):
        return [buildImage(y) for y in x]
    return (buildImageList(x) for x in idx)

data, meta, idx = dataLoad()
t_set, q_set, g_set = splitData(data, meta, idx)

print(t_set[0])