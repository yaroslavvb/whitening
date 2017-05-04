'''
CIFAR-10 example from https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
Now with weight normalization. Lines 64 and 69 contain the changes w.r.t. original.
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks
from keras import metrics
import os
import time
import argparse
import sys

from weightnorm import SGDWithWeightnorm
from weightnorm import AdamWithWeightnorm

import util as u
import numpy as np
import tensorflow as tf
import scipy

prefix = "keras_cifar_sgd"  # sgd wn
prefix = "keras_cifar_adam"  # adam wn
prefix = "keras_cifar_adamx4"  # adam wn with 4x larger bs
prefix = "keras_cifar_adamx16"  # adam wn with 16x larger bs
prefix = "keras_cifar_adamx256"  # adam wn with 64x larger bs

# second round of exp without shuffling
prefix = "keras_cifar2_adam"  # adam wn
prefix = "keras_cifar2_adamx4"  # adam wn with 4x larger bs
prefix = "keras_cifar2_adamx16"  # adam wn with 16x larger bs
prefix = "keras_cifar2_adamx256"  # adam wn with 64x larger bs

class TestCallback(callbacks.Callback):
  
  def __init__(self, data_train, data_test, fn):
    print("Creating callback")
    self.data_train = data_train
    self.data_test = data_test
    self.losses_train = []
    self.losses_test = []
    self.times = []
    self.fn = fn
    self.write_buffer = []
    self.last_save_ts = 0
    self.start_time = time.time()


  def on_epoch_end(self, epoch, logs={}):
    x_test, y_test = self.data_test
    x_train, y_train = self.data_train
    pixel_loss_test, acc1 = self.model.evaluate(x_test, y_test, verbose=0)
    pixel_loss_train, acc2 = self.model.evaluate(x_train, y_train, verbose=0)
    loss_test = pixel_loss_test*input_dim
    loss_train = pixel_loss_train*input_dim
    self.losses_train.append(loss_train)
    self.losses_test.append(loss_test)
    
    host = u.get_host_prefix()  # 10 for 10.cirrascale
    outfn = 'data/%s_%s'%(host, self.fn)

    if epoch == 0:
      os.system("rm -f "+outfn)
    elapsed = time.time()-self.start_time
    print('\n%d sec: Loss train: %.2f'%(elapsed,loss_train))
    print('%d sec: Loss test: %.2f'%(elapsed,loss_test))
    info_line = '%d, %f, %f, %f\n'%(epoch, elapsed, loss_train, loss_test)
    print(info_line[:-1])
    self.write_buffer.append(info_line)
    if time.time() - self.last_save_ts > 5*60:
      self.last_save_ts = time.time()
      with open(outfn, "a") as myfile:
        for line in self.write_buffer:
          myfile.write(line)
        self.write_buffer = []

          
if __name__=='__main__':
  np.random.seed(0)
  tf.set_random_seed(0)

  if prefix.endswith('x256'):
    batch_size = 32*64
  elif prefix.endswith('x16'):
    batch_size = 32*16
  elif prefix.endswith('x4'):
    batch_size = 32*4
  else:
    batch_size = 32*4
  nb_classes = 10
  nb_epoch = 20000
  data_augmentation = False

  # the data, shuffled and split between train and test sets
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()
  dsize = X_train.shape[0]
  print('X_train shape:', X_train.shape)
  print(X_train.shape[0], 'train samples')
  print(X_test.shape[0], 'test samples')
  X_train = X_train.astype('float32')
  print(X_train.shape)
  X_train = X_train.reshape((X_train.shape[0], -1))
  X_test = X_test.astype('float32')
  X_test = X_test.reshape((X_test.shape[0], -1))
  X_train /= 255
  X_test /= 255


  input_dim = 32*32*3
  model = Sequential()
  model.add(Dense(1024, input_dim=input_dim, activation='relu'))
  model.add(Dense(1024, activation='relu'))
  model.add(Dense(1024, activation='relu'))
  model.add(Dense(196, activation='relu'))
  model.add(Dense(1024, activation='relu'))
  model.add(Dense(1024, activation='relu'))
  model.add(Dense(1024, activation='relu'))
  model.add(Dense(input_dim, activation='relu'))

  #sgd_wn = SGDWithWeightnorm(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  sgd_wn = AdamWithWeightnorm()
  model.compile(loss='mean_squared_error',optimizer=sgd_wn,
                metrics=[metrics.mean_squared_error])
  cb = TestCallback((X_train, X_train), (X_test,X_test), prefix)

  # data based initialization of parameters
  from weightnorm import data_based_init
  data_based_init(model, X_train[:100])


  model.fit(X_train, X_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            validation_data=(X_test, X_test),
            callbacks=[cb],
            shuffle=False)
