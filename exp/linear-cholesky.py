#!/bin/env python -u
# Use orderered auto-regression of activations to obtain preconditioner
#
# Accompanying notebook:
# linear-cholesky.nb
# https://www.wolframcloud.com/objects/203cf855-c184-47e7-a778-bf24c7fb0411
#
# Need latest TF to run (for mat1 @ mat2 operator for matmul)
#
# pip install --upgrade https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-1.0.1-py3-none-any.whl


import numpy as np
import tensorflow as tf
import sys

def train_regular():
  tf.reset_default_graph()

  dtype = tf.float64
  XY0 = np.genfromtxt('linear-cholesky-XY.csv', delimiter= ",")
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  fsize = X0.shape[0]
  
  X_ = tf.placeholder(dtype, shape=X0.shape)
  Y_ = tf.placeholder(dtype, shape=Y0.shape)
  X = tf.Variable(X_, trainable=False)
  Y = tf.Variable(Y_, trainable=False)
  W = tf.Variable([[2, 1]], dtype=dtype)
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
  # {1.03475, 0.155254, 0.0270041, 0.00827915, 0.00552226, 0.00509353,
  # 0.00500441, 0.00496496, 0.00493292, 0.00490212}
  expected_losses = np.loadtxt("linear-cholesky-losses0.csv")
  np.testing.assert_allclose(expected_losses[:10],
                             observed_losses[:10], rtol=1e-12)


def train_cholesky():
  tf.reset_default_graph()

  dtype = tf.float64
  XY0 = np.genfromtxt('linear-cholesky-XY.csv', delimiter= ",")
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  fsize = X0.shape[0]
  
  X_ = tf.placeholder(dtype, shape=X0.shape)
  Y_ = tf.placeholder(dtype, shape=Y0.shape)
  X = tf.Variable(X_, trainable=False)
  Y = tf.Variable(Y_, trainable=False)
  W = tf.Variable([[2, 1]], dtype=dtype)
  error = tf.matmul(W, X)-Y   # 1 x d
  loss = tf.reduce_sum(tf.square(error))/dsize
  sess = tf.Session()

  cov_ = tf.matmul(X_, tf.transpose(X_))/dsize
  cov = tf.Variable(cov_, trainable=False)
  covI = tf.Variable(tf.matrix_inverse(cov_), trainable=False)

  init_dict = {X_: X0, Y_: Y0}
  lr = tf.constant(1, dtype=dtype)
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  
  (grad,var)  = opt.compute_gradients(loss, tf.trainable_variables())[0]
  ones_list = tf.constant([1.0]*fsize, dtype=dtype)
  
  # decomposition of covariance matrix
  Ch=tf.cholesky(cov)
  D1=tf.diag(tf.diag_part(Ch))
  L=Ch@tf.matrix_inverse(D1)
  cov2=L @ D1 @ D1 @ tf.transpose(L)

  # decomposition of precision matrix
  T = tf.matrix_inverse(L)
  covI2 = tf.transpose(T) @ tf.matrix_inverse(D1@D1) @ T

  # 1/2 of precision matrix recovers inverse Hessian
  pre = covI2/2

  grad = tf.matmul(grad, pre)
  train_op = opt.apply_gradients([(grad, var)])

  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  assert sess.run(tf.norm(cov2-cov))==0.
  assert sess.run(tf.norm(covI2-covI))==0.

  observed_losses = []
  for i in range(100):
    loss0, _ = sess.run([loss, train_op])
    observed_losses.append(loss0)

  # from accompanying notebook
  # [1.03475, 4.90422*10^-28, 0., 0., 0., 0., 0., 0., 0., 0....
  
  expected_losses = np.loadtxt("linear-cholesky-losses1.csv")
  np.testing.assert_allclose(expected_losses[:10],
                             observed_losses[:10], rtol=1e-12, atol=1e-12)



if __name__ == '__main__':
  train_regular()
  train_cholesky()
  print("All tests passed")
