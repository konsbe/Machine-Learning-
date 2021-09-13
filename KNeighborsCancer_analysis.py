# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:08:08 2021

@author: konsb
"""

import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mglearn
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt



mglearn.plots.plot_knn_classification(n_neighbors = 1)
plt.show()


mglearn.plots.plot_knn_classification(n_neighbors = 3)
plt.show()

diabetes = pd.read_csv('diabetes.csv')

print(diabetes.head())
print(diabetes.shape) #(768,9)

X = np.array(diabetes['Outcome'])
x = np.array(diabetes.drop('Outcome', axis = 1))


y = np.array(diabetes['Outcome'])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 0)

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

math.sqrt(len(y_test))  #Out[10]: 12.409673645990857

classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'euclidean')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)



best_knn = accuracy_score(y_test, y_pred)
knn = 1
j = 2
while j < 16:
    classifier = KNeighborsClassifier(n_neighbors = j, metric = 'euclidean')
    classifier.fit(x_train, y_train)
    
    y_pred = classifier.predict(x_test)
#    print(accuracy_score(y_test, y_pred))
    j+=1
    metric = accuracy_score(y_test, y_pred)
    if metric > best_knn:
        best_knn = accuracy_score(y_test, y_pred)
        knn = j

print(best_knn, knn)    #0.8051948051948052 k= 7 random_state = 0
                        #0.7532467532467533 k= 16 random_state = 45





