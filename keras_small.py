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

optimizer = sys.argv[1]
epochs = int(sys.argv[2])
# Run comparison with Keras optimizer

prefix = "100_"+sys.argv[1]  # raise epochs
batch_size=100

prefix = "10000_"+sys.argv[1]
batch_size=10000


from keras import optimizers
import load_MNIST

from keras.models import Sequential
from keras.layers import Dense
import numpy

#
if __name__=='__main__':

  numpy.random.seed(0)
  
  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  dsize = 10000
  X = train_images[:,:dsize].T

  if optimizer=='sgd':
    optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

  elif optimizer=='sgd_wn':
    optimizer = SGDWithWeightnorm(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  elif optimizer=='adam':
    optimizer = optimizers.Adam()
  elif optimizer =='adam_wn':
    print("Creating adam weight norm optimizer")
    optimizer = AdamWithWeightnorm()
    

  model = Sequential()
  model.add(Dense(196, input_dim=28*28, activation='relu'))
  model.add(Dense(28*28, activation='relu'))
  model.compile(loss='mean_squared_error', optimizer=optimizer,
                metrics=[metrics.mean_squared_error])
  # nb_epochs in older version
  result = model.fit(X, X, batch_size=batch_size, nb_epoch=epochs) # epochs=100, 
  acc_hist = np.asarray(result.history['mean_squared_error'])*28*28  # avg pixel loss->avg image image loss
  u.dump(acc_hist, "%s_losses.csv"%(prefix,))

