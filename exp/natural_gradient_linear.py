#!/bin/env python -u
#
# Apply natural gradient (using empirical Fisher) to a small
# overparameterizerd inference problem.
#
# Accompanying notebook:
# natural_gradient_linear.nb
# https://www.wolframcloud.com/objects/492a0bf1-2282-4e16-9e6b-1e9450412269

import numpy as np
import tensorflow as tf
import sys

dtype = np.float64

def pseudo_inverse(mat):
  s, u, v = tf.svd(mat)
  eps = 1e-10   # zero threshold for eigenvalues
  si = tf.where(tf.less(s, eps), s, 1./s)
  return u @ tf.diag(si) @ tf.transpose(v)

def train_natural():
  tf.reset_default_graph()

  # global settings
  W0 = np.array([[2, 1, 0]], dtype=dtype)
  lr = 0.5
  
  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_linear_data.csv', delimiter= ",")
  X0 = XY0[:-1,:]  # 3 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  X_ = tf.placeholder(dtype, shape=X0.shape)
  Y_ = tf.placeholder(dtype, shape=Y0.shape)
  X = tf.Variable(X_, trainable=False)
  Y = tf.Variable(Y_, trainable=False)
  init_dict={X_: X0, Y_: Y0}
  
  W = tf.Variable(W0, dtype=dtype)
  error = Y - W@X   # 1 x d
  loss = tf.reduce_sum(tf.square(error))/dsize

  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  (grad, var)  = opt.compute_gradients(loss, tf.trainable_variables())[0]

  # empirical fisher is covariance matrix weighted by error of each datapoint
  X2 = X*error
  fisher = X2 @ tf.transpose(X2) / dsize
  pre = pseudo_inverse(fisher)
  train_op = opt.apply_gradients([(grad @ pre, var)])

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
    
  observed_losses = []
  for i in range(25):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op)

  # from accompanying notebook
  # {1.03475, 0.501415, 0.077998, 0.636779, 0.1772, 0.0862901, 0.533846, \
  # 0.101634

  expected_losses = np.loadtxt("data/natural_gradient_linear_losses.csv")
  np.testing.assert_allclose(expected_losses[:10],
                             observed_losses[:10], rtol=1e-9, atol=1e-20)
  

if __name__ == '__main__':
  train_natural()
