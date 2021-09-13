# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:19:58 2021

@author: konsb
"""

from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np


(train_imgs, train_lbls), (test_imgs, test_lbls) = mnist.load_data()


train_imgs = train_imgs.reshape((60000, 28, 28, 1))
train_imgs = train_imgs.astype('float32') / 255

test_imgs = test_imgs.reshape((10000, 28, 28, 1))
test_imgs = test_imgs.astype('float32') / 255

train_lbls = to_categorical(train_lbls)
test_lbls = to_categorical(test_lbls)

train_lbls
test_lbls

model = models.Sequential()
model.add(layers.Conv2D (32, (3,3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'valid'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))


model.summary()

model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy', metrics=['accuracy'])

model.fit(train_imgs, train_lbls,
          batch_size= 64, epochs= 5)

    
test_loss, test_acc = model.evaluate(test_imgs, test_lbls)
print(f'accuracy test = {test_acc}')        #accuracy test = 0.9905999898910522


# testing our model checking the values and images 
predictions = model.predict(x = test_imgs, batch_size = 10, verbose = 0)
for p in predictions:
    print(np.argmax(p))


print(np.argmax(predictions[0]))
plt.imshow(test_imgs[0])
plt.show