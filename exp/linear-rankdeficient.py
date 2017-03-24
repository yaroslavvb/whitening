#!/bin/env python -u
# Try regular SGD and SGD with pseudo-inverse preconditioner on a deficient
# linear regression problem
#
# Accompanying notebook:
# linear-rankdeficient-join.nb
# https://www.wolframcloud.com/objects/3ebc256e-dc25-4513-971e-082d28fae38a
# Fixed data (centered X's + losses) generated from
# linear-cholesky-rankdeficient.nb
#
# Additional sanity checks in:
# linear-rankdeficient.nb


import numpy as np
import tensorflow as tf
import sys

def train_regular():
  tf.reset_default_graph()

  dtype = tf.float64
  XY0 = np.genfromtxt('linear-rankdeficient.csv', delimiter= ",")
  X0 = XY0[:-1,:]  # 3 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  fsize = X0.shape[0]
  
  X_ = tf.placeholder(dtype, shape=X0.shape)
  Y_ = tf.placeholder(dtype, shape=Y0.shape)
  X = tf.Variable(X_, trainable=False)
  Y = tf.Variable(Y_, trainable=False)
  W = tf.Variable([[2, 1, 0]], dtype=dtype)
  error = tf.matmul(W, X)-Y   # 1 x d
  loss = tf.reduce_sum(tf.square(error))/dsize
  sess = tf.Session()

  cov_ = tf.matmul(X_, tf.transpose(X_))/dsize
  cov = tf.Variable(cov_, trainable=False)
  covI = tf.Variable(tf.matrix_inverse(cov_), trainable=False)

  init_dict = {X_: X0, Y_: Y0}
  lr = tf.constant(.15, dtype=dtype)
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  
  (grad,var)  = opt.compute_gradients(loss, tf.trainable_variables())[0]
  ones_list = tf.constant([1.0]*fsize, dtype=dtype)
  pre = tf.diag(ones_list)
    
  grad = tf.matmul(grad, pre)
  train_op = opt.apply_gradients([(grad, var)])

  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

  observed_losses = []
  for i in range(100):
    loss0, _ = sess.run([loss, train_op])
    observed_losses.append(loss0)

  # from accompanying notebook
  # [1.04094, 0.781092, 0.586415, 0.440565, 0.331294, 0.249426, 0.188086,
  #  0.142126, 0.107688 ...
  expected_losses = np.loadtxt("linear-rankdeficient-losses.csv")
  np.testing.assert_allclose(expected_losses[:10],
                             observed_losses[:10], rtol=1e-12)


def train_pseudoinverse():
  tf.reset_default_graph()

  dtype = tf.float64
  XY0 = np.genfromtxt('linear-rankdeficient-fixed.csv', delimiter= ",")
  X0 = XY0[:-1,:]  # 3 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  fsize = X0.shape[0]
  
  X_ = tf.placeholder(dtype, shape=X0.shape)
  Y_ = tf.placeholder(dtype, shape=Y0.shape)
  X = tf.Variable(X_, trainable=False)
  Y = tf.Variable(Y_, trainable=False)
  W = tf.Variable([[2, 1, 0]], dtype=dtype)
  error = tf.matmul(W, X)-Y   # 1 x d
  loss = tf.reduce_sum(tf.square(error))/dsize
  sess = tf.Session()

  cov_ = tf.matmul(X_, tf.transpose(X_))/dsize
  cov = tf.Variable(cov_, trainable=False)
  covI = tf.Variable(tf.matrix_inverse(cov_), trainable=False)

  init_dict = {X_: X0, Y_: Y0}
  lr = tf.constant(.5, dtype=dtype)
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  
  (grad,var)  = opt.compute_gradients(loss, tf.trainable_variables())[0]
  ones_list = tf.constant([1.0]*fsize, dtype=dtype)

  # pseudo-inverse preconditioner
  s, u, v = tf.svd(cov)
  eps = 1e-10   # threshold used to determine which singular values are zero
  si = tf.where(tf.less(s, eps), s, 1./s)
  pre = u @ tf.diag(si) @ tf.transpose(v)

  grad = grad @ pre
  train_op = opt.apply_gradients([(grad, var)])

  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  observed_losses = []
  for i in range(100):
    loss0, _ = sess.run([loss, train_op])
    observed_losses.append(loss0)

  # from accompanying notebook
  # [1.04094, 0.510063, 0.249931, 0.122466, 0.0600084, 0.0294041,
  #  0.014408, 0.00705992, 0.00345936, 0.00169509, 0.000830593...
  expected_losses = np.loadtxt("linear-rankdeficient-losses-pre-fixed.csv")
  np.testing.assert_allclose(expected_losses[:10],
                            observed_losses[:10], rtol=1e-10, atol=1e-20)


if __name__ == '__main__':
  #train_regular()
  train_pseudoinverse()
