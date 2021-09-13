    # -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:09:18 2021

@author: konsb
"""

import os

print(os.path)      #<module 'ntpath' from 'C:\\Users\\konsb\\anaconda3\\lib\\ntpath.py'>
print(os.getcwd())  #C:\Users\konsb


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle
from matplotlib import style


temps = pd.read_csv('temps.csv')

print(temps.head(5))

temps = pd.get_dummies(temps)
temper = temps

print(temps.shape)

labels = np.array(temps['actual']) #predicting values
x = labels
temps = temps.drop('actual', axis = 1)

temps_list = list(temps.columns)
temps = np.array(temps)

x_train, x_test, y_train, y_test = train_test_split(temps, labels, test_size = 0.2,
                                                    random_state = 45)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

predicts = x_test[:, temps_list.index('average')]
predicts_error = abs(predicts - y_test)

print("avg error:", round(np.mean(predicts_error), 2), "oC") #avg error: 3.97 oC

#model training
rf = RandomForestRegressor(n_estimators=1000, random_state = 45)
rf.fit(x_train, y_train)

predictions = rf.predict(x_test)
error = abs(predictions - y_test)

print('Mean Absolute Error:', round(np.mean(predicts_error), 2), "oC") 
#Mean Absolute Error: 3.97 oC

mape = 100 * (error / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy = ', round(accuracy, 2), "%") #Accuracy =  94.44 %


#saving our model
with open('temp_model.pickle', 'wb') as f:
    pickle.dump(RandomForestRegressor, f)

#load our model wihout running the whole code
pickle_in = open('temp_model.pickle', 'rb')
rf = pickle.load(pickle_in)


#run and checking our model (comparing with the actual values)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])



#print(temper.head(5))

#correlation between data points
style.use('ggplot')
plt.scatter(temper['friend'], temper['actual'])
plt.xlabel(temper['friend'])
plt.ylabel(temper['average'])
plt.show()


