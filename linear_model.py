# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:32:53 2021

@author: konsb
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv('temps.csv')

print(data.head(5))

data = pd.get_dummies(data)
#print(data.shape)


#data = data.drop('actual', axis = 1)
#print(data.shape)

x = np.array(data.drop('actual',axis = 1))
y = np.array(data['actual'])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print('Accurancy:', acc)      #Accurancy: 0.8828483755354554

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


