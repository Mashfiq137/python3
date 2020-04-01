# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:34:30 2020

@author: Niloy
"""

#utf general ci


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset.csv',error_bad_lines=False)
X=dataset.loc[:,['latitude1','longitude1']]

# Finding out the optimal number of clusters:
# We are using "elbow method"

from sklearn.cluster import KMeans
# Plotting graph:
wcss = []
for i in range(1,2):
    kmeans = KMeans(n_clusters=i, init='k-means++',max_iter = 300,n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,2),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to the null dataset:

kmeans = KMeans(n_clusters = 2,init = 'k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)
X =np.array(X)
# Visualizing the clusters:

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='magenta',label='Cluster 2')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of Accidents')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
