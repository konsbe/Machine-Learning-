# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 18:32:14 2021

@author: konsb
"""



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

