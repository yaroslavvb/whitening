#!/bin/env python -u
# Try inverse covariance preconditioner on a simple linear regression problem
#
# Accompanying notebook:
# https://www.wolframcloud.com/objects/76c60b54-1f01-4139-8a92-f0bdaa2ab064

import numpy as np
import tensorflow as tf


def train(preconditioner):
  tf.reset_default_graph()

  XY0 = np.genfromtxt('linear.csv', delimiter= ",")
  X0 = XY0[:2,:]  # 2 x d
  Y0 = XY0[2:,:]  # 1 x d
  dsize = X0.shape[1]
  
  X_ = tf.placeholder(tf.float32, shape=X0.shape)
  Y_ = tf.placeholder(tf.float32, shape=Y0.shape)
  X = tf.Variable(X_, trainable=False)
  Y = tf.Variable(Y_, trainable=False)
  W = tf.Variable([[2, 1]], dtype=np.float32)
  error = tf.matmul(W, X)-Y   # 1 x d
  loss = tf.reduce_sum(tf.square(error))
  sess = tf.Session()

  cov_ = tf.matmul(X_, tf.transpose(X_))/dsize
  cov = tf.Variable(cov_, trainable=False)
  covI = tf.Variable(tf.matrix_inverse(cov_), trainable=False)

  init_dict = {X_: X0, Y_: Y0}
  lr = .1/dsize
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  
  (grad,var)  = opt.compute_gradients(loss, tf.trainable_variables())[0]
  if preconditioner == 'identity':
    pre = tf.diag([1., 1.])
  elif preconditioner == 'cov': # inverse of empirical covariance
    pre = covI
  else:
    assert False, "Unknown preconditioner"
    
  grad = tf.matmul(grad, pre)
  train_op = opt.apply_gradients([(grad, var)])

  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

  for i in range(100):
    loss0, _ = sess.run([loss, train_op])
    print(loss0)

if __name__ == '__main__':
  print("Training with no preconditioner")
  train("identity")
  print("Training with inverse covariance preconditioner")
  train("cov")
