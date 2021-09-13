# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 18:39:35 2021

@author: konsb

Here we are performing 4 different ways to score our accuracy in our model
As we see in this dataset the first model with svm and the third model give us the 
best results to make predictions,

"""

""" FIRST WAY """

import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)


x =  cancer.data
y = cancer.target

classes = ['malignant', 'benign']

x_train, x_test, y_train, y_test = (
    sklearn.model_selection.train_test_split(x, y, test_size=0.3)) 


print (x_train, y_train)

clf = svm.SVC(kernel = 'linear') #support vector classification
clf.fit(x_train, y_train) #fit them

y_pred = clf.predict(x_test) #making predictions in some data

acc = metrics.accuracy_score(y_test, y_pred) #accurancy

print(acc) #>>OUTPUT = 0.8771929824561403 - 0.9122807017543859 
                #better than the expected with SVC kernel as default
                    
                    #with SVC kernel set to linear 
                    #>>output =0.9298245614035088 - 0.9707602339181286
 
"""SECOND WAY"""


#from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

classes = ['malignant', 'benign']

cancer = datasets.load_breast_cancer()
x =  cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
svc = SVC()
svc.fit(x_train, y_train)


print(f'Training Accuracy: {svc.score(x_train, y_train)}',
      f'\nTest Accuracy: {svc.score(x_test, y_test)}')


# =============================================================================
# >>OUTPUT Training Accuracy: 0.903755868544601 
#             Test Accuracy: 0.9370629370629371
# =============================================================================

"""THIRD WAY"""

min_train = x_train.min(axis = 0)
train_range = (x_train - min_train).max(axis = 0)

#print(min_train,"\n",train_range)


x_train_scale = (x_train - min_train) / train_range #by deviding we scale the numbers between
x_test_scale = (x_test - min_train) / train_range   #0-1 for having a better accuracy

#print(x_train_scale,"\n",x_test_scale)
svc = SVC()
svc.fit(x_train_scale, y_train)

print(f'Training Accuracy: {svc.score(x_train_scale, y_train)}',
      f'\nTest Accuracy: {svc.score(x_test_scale, y_test)}')

# =============================================================================
# >>OUTPUT Training Accuracy: 0.9835680751173709 
#             Test Accuracy: 0.972027972027972
# =============================================================================


"""FOURTH WAY WORKING WITH KNeighborsClassifier"""

cancer = datasets.load_breast_cancer()

x =  cancer.data
y = cancer.target


x_train, x_test, y_train, y_test = (
    sklearn.model_selection.train_test_split(x, y, test_size=0.3)) 

classes = ['malignant', 'benign']
j = 1
while j < 14:
    clf = KNeighborsClassifier(n_neighbors = j)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    acc = metrics.accuracy_score(y_test, y_pred)
    print(acc)
    j+=1
    
    
# =============================================================================
# >> OUTPUT 0.9298245614035088
#             0.9239766081871345
#             0.9473684210526315
#             0.9415204678362573
#             0.9532163742690059  
#             0.9590643274853801
#             0.9415204678362573
#             0.9532163742690059
#             0.9473684210526315
#             0.9532163742690059
#             0.9473684210526315
#             0.9473684210526315
#             0.935672514619883    
#   we are not having the same accuracy every time we perform our model
#   mostly as the n_nighbors rises the accuracy rises too
# =============================================================================
