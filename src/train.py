# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:55:38 2018

@author: zw4215
"""

from dataproc import toFeatureArray, toLabelArray

# ------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------

def train(clf, t_set, q_set, g_set):
    t_f = clf.fit_transform(toFeatureArray(t_set), toLabelArray(t_set))
    #ratio = pca.explained_variance_ratio_
    q_f = clf.transform(toFeatureArray(q_set))
    g_f = clf.transform(toFeatureArray(g_set))
    for i, t_img in enumerate(t_set):
        t_img.feature = t_f[i]
    for i, q_img in enumerate(q_set):
        q_img.feature = q_f[i]
    for i, g_img in enumerate(g_set):
        g_img.feature = g_f[i]
    return None

def unsup_transform(clf, t_set, q_set, g_set):
    t_f = clf.fit_transform(toFeatureArray(t_set))
    q_f = clf.fit_transform(toFeatureArray(q_set))
    g_f = clf.fit_transform(toFeatureArray(g_set))
    for i, t_img in enumerate(t_set):
        t_img.feature = t_f[i]
    for i, q_img in enumerate(q_set):
        q_img.feature = q_f[i]
    for i, g_img in enumerate(g_set):
        g_img.feature = g_f[i]
    return None

def train_rca(rca, chunks, t_set, q_set, g_set):
    rca.fit(toFeatureArray(t_set), chunks)
    t_f = rca.transform(toFeatureArray(t_set))
    q_f = rca.transform(toFeatureArray(q_set))
    g_f = rca.transform(toFeatureArray(g_set))
    for i, t_img in enumerate(t_set):
        t_img.feature = t_f[i]
    for i, q_img in enumerate(q_set):
        q_img.feature = q_f[i]
    for i, g_img in enumerate(g_set):
        g_img.feature = g_f[i]
    return None

# This one is shit. REVISIT: fix
#import sklearn.manifold as mnf
#mds = mnf.MDS(n_components=3, metric=True, n_init=3, max_iter=30, verbose=2, eps=0.001, n_jobs=None, random_state=None, dissimilarity='euclidean')
#q_set_mds_feature = mds.fit_transform(toFeatureArray(q_set))
#g_set_mds_feature = mds.fit_transform(toFeatureArray(g_set))
#for i, q_img in enumerate(q_set):
#    q_img.feature = q_set_mds_feature[i]
#for i, g_img in enumerate(g_set):
#    g_img.feature = g_set_mds_feature[i]
#del q_set_mds_feature, g_set_mds_feature
#
#lap('Train with MDS', tr)