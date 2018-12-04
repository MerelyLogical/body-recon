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

# ------------------------------------------------------------------------------
# Data Structure
# ------------------------------------------------------------------------------
class Image(object):
    'storing all information related to a single data point'
    def __init__ (self, feature, camId, path, label):
        self.feature = feature
        self.camId = camId
        self.path = path
        self.label = label
        
    def __str__ (self):
        'allows printing'
        display = 'Class: {}; Camera: {}; Path: {}'\
            .format(self.label, self.camId, self.path)
        return display
    
    def subplot (self):
        'plots self'
        PATH = '../data/images_cuhk03/'
        plt.imshow(mpimg.imread(PATH + self.path))
        plt.title('Class: {}'.format(self.label))
        return None
        
def toFeatureArray(images):
    return np.asarray ([image.feature for image in images])

def toLabelArray(images):
    return np.asarray ([image.label for image in images])

# ------------------------------------------------------------------------------
# Loading Data
# ------------------------------------------------------------------------------
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