#!/bin/env python -u
#
# Apply natural gradient (using empirical Fisher) to a multilayer problem
#

import numpy as np
import tensorflow as tf
import sys

dtype = np.float64

def pseudo_inverse(mat):
  s, u, v = tf.svd(mat)
  eps = 1e-10   # zero threshold for eigenvalues
  si = tf.where(tf.less(s, eps), s, 1./s)
  return u @ tf.diag(si) @ tf.transpose(v)

def identity(n):
  return tf.diag(tf.ones((n,), dtype=dtype))


# partitions numpy array into sublists of given sizes
def partition(vec, sizes):
  assert np.sum(sizes) == len(vec)
  splits = []
  current_idx = 0
  for i in range(len(sizes)):
    splits.append(vec[current_idx: current_idx+sizes[i]])
    current_idx += sizes[i]
  assert current_idx == len(vec)
  return splits

  
# turns flattened representation into list of matrices with given matrix
# sizes
def unflatten(Wf, fs):
  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  assert np.sum(sizes)==len(Wf)
  Wsf = partition(Wf, sizes)
  Ws = [unvectorize(Wsf[i], dims[i][0]) for i in range(len(sizes))]
  return Ws


def unvectorize(vec, rows):
  """Turns vectorized version of tensor into original matrix with given
  number of rows."""
  assert len(vec)%rows==0
  cols = len(vec)//rows;
  return np.array(np.split(vec, cols)).T


def check_equal(a, b, rtol=1e-12, atol=1e-12):
  np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


# convention, X0 is numpy, X is Tensor
def train_regular():
  """Train network, with explicit backprop."""
  
  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                     delimiter= ",")
  W0s = unflatten(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  W = []   # list of "W" matrices. W[0] is input matrix (X), W[n] is last matrix
  Wi_holders = []
  A = [identity(dsize)]   # activation matrices
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    Wi_name = "W"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_holder = tf.placeholder(dtype, shape=Wi_shape, name=Wi_name+"_h")
    Wi_holders.append(Wi_holder)  # TODO: delete
    Wi = tf.Variable(Wi_holder, name=Wi_name, trainable=(i>0))
    Ai_name = "A"+str(i+1)
    print("Multiplying %s and %s " %(Wi.shape, A[-1].shape))
    Ai = tf.matmul(Wi, A[-1], name=Ai_name)
    A.append(Ai)
    W.append(Wi)
    
    init_dict[Wi_holder] = W0s[i]

  assert len(A) == n+2
  
  assert W[0].shape == (2, 10)
  assert W[1].shape == (2, 2)
  assert W[2].shape == (2, 2)
  assert W[3].shape == (1, 2)

  assert X0.shape == (2, 10)
  assert W0s[1].shape == (2, 2)
  assert W0s[2].shape == (2, 2)
  assert W0s[3].shape == (1, 2)
  
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = 0.5
  
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  grads_and_vars = opt.compute_gradients(loss, tf.trainable_variables())
  train_op = opt.apply_gradients(grads_and_vars)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/natural_gradient_multilayer_losses_regular.csv")
  
  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op)

  check_equal(observed_losses, expected_losses)

if __name__ == '__main__':
  train_regular()
