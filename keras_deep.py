#!/usr/bin/env python
# Run comparison with Keras optimizer

from keras import optimizers
from weightnorm import SGDWithWeightnorm
from weightnorm import AdamWithWeightnorm

import load_MNIST
import util as u
import numpy as np
import scipy

from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import numpy
import sys

#prefix = "100_"+sys.argv[1]  # raise epochs
batch_size=10000
prefix = "keras_deep_long"
prefix = "keras_deep_long_full"  # large batch
epochs = 20000 # 2 sec per epoch, 12 hours




from keras import optimizers
import load_MNIST

from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks
import numpy


class TestCallback(callbacks.Callback):
  def __init__(self, test_data):
    print("Creating callback")
    self.test_data = test_data
    self.losses = []

  def on_epoch_end(self, epoch, logs={}):
    x, y = self.test_data
    pixel_loss, acc = self.model.evaluate(x, y, verbose=0)
    loss = pixel_loss*28*28
    self.losses.append(loss)
    print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

if __name__=='__main__':
  numpy.random.seed(0)
  
  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  dsize = 10000
  X = train_images[:,:dsize].T
  X_test = train_images[:,-dsize:].T
  
  optimizer = AdamWithWeightnorm()

  model = Sequential()
  model.add(Dense(196, input_dim=28*28, activation='relu'))
  model.add(Dense(98, activation='relu'))
  model.add(Dense(49, activation='relu'))
  model.add(Dense(98, activation='relu'))
  model.add(Dense(196, activation='relu'))
  model.add(Dense(28*28, activation='relu'))
  model.compile(loss='mean_squared_error', optimizer=optimizer,
                metrics=[metrics.mean_squared_error])
  # nb_epochs in older version
  cb = TestCallback((X_test,X_test))
  
  result = model.fit(X, X, validation_data=(X_test, X_test), 
                     batch_size=batch_size,
                     nb_epoch=epochs,
                     callbacks=[cb])

  acc_hist = np.asarray(result.history['mean_squared_error'])*28*28  # avg pixel loss->avg image image loss
  u.dump(acc_hist, "%s_losses.csv"%(prefix,))
  u.dump(cb.losses, "%s_vlosses.csv"%(prefix,))

