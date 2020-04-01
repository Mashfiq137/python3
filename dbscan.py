# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:36:12 2020

@author: Niloy
"""

#DB SCAN clustering
#essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#import dataset
dataset = pd.read_csv('dataset.csv',error_bad_lines=False)
X=dataset.loc[:,['latitude1','longitude1']]

#import dbscan
from sklearn.cluster import DBSCAN
dbscan= DBSCAN(eps = 5, min_samples = 5)


model = dbscan.fit(X)
labels = model.labels_


from sklearn import metrics

sample_cores = np.zeros_like(labels,dtype = bool)
sample_cores[dbscan.core_sample_indices_] = True


n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(metrics.silhouette_score(X,labels))
