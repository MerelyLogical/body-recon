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
import time
from sklearn import neighbors

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
        PATH = '../data/images_cuhk03/'
        plt.imshow(mpimg.imread(PATH + self.path))
        display = 'Class: {}\nCamera: {}\nPath: {}'\
            .format(self.label, self.camId, self.path)
        return display

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

# ------------------------------------------------------------------------------
# Nearest Neighbour
# ------------------------------------------------------------------------------
def neighbor(query, gallery, k):
    'finds indexes of the k nearest neighbours in gallery'
    clf = neighbors.KNeighborsClassifier()
    clf.fit(toFeatureArray(gallery), toLabelArray(gallery))
    return clf.kneighbors(toFeatureArray(query), k)

# ------------------------------------------------------------------------------
# Performance
# ------------------------------------------------------------------------------
def lap(event, records):
    'stopwatch'
    t = time.perf_counter()
    print ('[Timer] {} took {}s, total time {}s'\
           .format(event, t-records[-1], t-records[0]))
    records.append(t)
    return None

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
tr = [time.perf_counter()]

lap('Start', tr)

data, meta, idx = dataLoad()
t_set, q_set, g_set = splitData(data, meta, idx)
del data, meta, idx

lap('Load data', tr)

print (t_set[0])

lap('Print data', tr)

q_neighbor = neighbor(q_set, g_set, 10)

lap('Calculate 10-NN', tr)
