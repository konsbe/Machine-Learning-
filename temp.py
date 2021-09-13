# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

l1 = pd.Series([5,8,3,6,5], ['cow','dog','fox', 'chicken', 'cat'])

df = pd.DataFrame([l1])

#print(df)

l2 = pd.Series([15,82,3,61,3], ['cow','dog','fox', 'chicken', 'cat'])

index = 'Row1 Row2'.split()

df1 = pd.DataFrame([l1,l2], index)

#print(type(df1))
print(df1)





from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (16,9)

X,y = make_blobs(n_samples= 800, n_features = 3, centers = 4)

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:, 1], X[:, 2] )


kmeans = KMeans(n_clusters=4)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = y)

ax.scatter(C[:, 0], X[:, 1], X[:, 2], marker='*', c='#050505', s=1000)



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import time 

start = time.time()
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
#print(mnist.data.shape)

from sklearn.datasets import make_classification
x, y = make_classification(random_state=42)


x_train, x_test, y_train, y_test = (
    train_test_split(mnist.data, 
                     mnist.target, 
                     test_size = 1/7.0, random_state = None)) #random_state None is the default
                                                                #and we getting different values
#print(x_train)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(.95)

pca.fit(x_train)

#print(pca.n_components_)

x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

#max_iter default value 1000. Increasing the ierations in this model we getting more sagnificant results
#increasing iterations = increasing train time

logisticRegr = LogisticRegression(solver = 'lbfgs', max_iter=2000) 

logisticRegr.fit(x_train, y_train)
logisticRegr.score(x_test, y_test)

end = time.time()
proc_time = end - start

print(f'accurancy = {logisticRegr.score(x_test, y_test)}', f"\ntime processing = {proc_time}")


# =============================================================================
# >>output
# accurancy = 0.9218 
# time processing = 99.02086448669434
# =============================================================================






