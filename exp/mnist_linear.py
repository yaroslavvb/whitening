#!/bin/env python -u
# Minimize linear model on MNIST
# Check that K-Fac/Regression/Newton's method produce same result
#


import numpy as np
import tensorflow as tf
import sys

def test_sgd():
  tf.reset_default_graph()

  dtype = np.float64
  XY0 = np.genfromtxt('mnist_small.csv', delimiter= ",")
  f2 = 10 # number of classes
  
  X0 = XY0[:-f2,:]  # 50 x d
  Y0 = XY0[-f2:,:]  # 10 x d
  dsize = X0.shape[1]
  f1 = X0.shape[0]
  assert f1 == 50
  
  X_ = tf.placeholder(dtype, shape=X0.shape)
  Y_ = tf.placeholder(dtype, shape=Y0.shape)
  X = tf.Variable(X_, trainable=False)
  Y = tf.Variable(Y_, trainable=False)

  # initial W0 predicts 0.1 for every instance for every class
  W0 = np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T
  W = tf.Variable(W0.astype(dtype))
  
  error = W @ X - Y   # 10 x d
  
  #  Use algebra notation, equivalent to tf.reduce_sum(error*error)/dsize
  loss = tf.trace(error @ tf.transpose(error))/dsize

  sess = tf.InteractiveSession()

  # cov_ = tf.matmul(X_, tf.transpose(X_))/dsize
  # cov = tf.Variable(cov_, trainable=False)
  # covI = tf.Variable(tf.matrix_inverse(cov_), trainable=False)

  init_dict = {X_: X0, Y_: Y0}
  lr = tf.constant(1e-6, dtype=dtype)
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  
  (grad, var)  = opt.compute_gradients(loss, tf.trainable_variables())[0]
  ones_list = tf.constant([1.0]*f2, dtype=dtype)
  
  pre = tf.diag(ones_list)  # f2 x f2
  grad = pre @ grad
  train_op = opt.apply_gradients([(grad, var)])

  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  assert(abs(loss.eval()-0.9) < 1e-15)
  
  observed_losses = []
  for i in range(100):
    loss0, _ = sess.run([loss, train_op])
    observed_losses.append(loss0)

  # from accompanying notebook
  # {0.9, 0.888968, 0.878518, 0.868525, 0.858921, 0.849668, ...
  expected_losses = np.loadtxt("mnist_linear_losses0.csv")
  np.testing.assert_allclose(expected_losses[:20],
                             observed_losses[:20], rtol=1e-12)


def test_newton():
  pass
  
if __name__ == '__main__':
  test_sgd()
  print("All tests passed")
  
