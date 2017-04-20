# KFAC implementation and test on MNIST

import networkx as nx
import load_MNIST
import numpy as np
import scipy.io # for loadmat
from scipy import linalg # for svd
import matplotlib # for matplotlib.cm.gray
from matplotlib.pyplot import imshow
import math
import time

import os, sys
use_gpu = False
if use_gpu:
  os.environ['CUDA_VISIBLE_DEVICES']='0'
else:
  os.environ['CUDA_VISIBLE_DEVICES']=''

import tensorflow as tf
import util
import util as u
from util import t  # transpose

import load_MNIST


def W_uniform(s1, s2):
  r = np.sqrt(6) / np.sqrt(s1 + s2 + 1)
  result = np.random.random(2*s2*s1)*2*r-r
  return result


if __name__ == '__main__':

  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  dsize = 10000
  patches = train_images[:,:dsize];
  X0=patches
  fs = [dsize, 28*28, 196, 28*28]
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = f(-1)
  n = len(fs) - 2

  eval_batch_size = 100

  np.random.seed(0)
  tf.set_random_seed(0)
  
  dtype = np.float32
  # 64-bit doesn't help much, see
  # https://www.wolframcloud.com/objects/5f297f41-30f7-4b1b-972c-cac8d1f8d8e4
  u.default_dtype = dtype
  machine_epsilon = np.finfo(dtype).eps  # TODO: tie it to dtype

  # helper to initialize variables, adds initial values to init_dict
  init_dict = {}     # {var_placeholder: init_value}
  vard = {}      # {var: VarInfo}
  def init_var(val, name, trainable=False, noinit=False):
    if isinstance(val, tf.Tensor):
      collections = [] if noinit else None
      var = tf.Variable(val, name=name, collections=collections)
    else:
      val = np.array(val)
      assert u.is_numeric, "Unknown type"
      holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
      var = tf.Variable(holder, name=name, trainable=trainable)
      init_dict[holder] = val
    var_p = tf.placeholder(var.dtype, var.shape)
    var_setter = var.assign(var_p)
    vard[var] = u.VarInfo(var_setter, var_p)
    return var

  train_data_node = tf.placeholder(dtype, shape=(dsize, 28, 28, 1))
  eval_data = tf.placeholder(dtype, shape=(eval_batch_size, 28, 28, 1))

  W0f = W_uniform(fs[2],fs[3]).astype(dtype)
  W0 = u.unflatten(W0f, fs[1:])

  X = init_var(X0, "X")
  W = [X]
  for layer in range(1, n+1):
    W.append(init_var(W0[layer-1], "W"+str(layer)))

  def sigmoid(x): return tf.sigmoid(x)

  A = [None]*(n+2)
  with tf.control_dependencies([tf.assert_equal(1, 0, message="too huge")]):
    A[0] = u.Identity(dsize, dtype=dtype)
  A[1] = W[0]
  for i in range(1, n+1):
    A[i+1] = sigmoid(W[i] @ A[i])
    
  err = (A[3] - A[1])
  loss = u.L2(err) / (2 * dsize)

  optimizer = tf.train.GradientDescentOptimizer(0.2)
  train_op = optimizer.minimize(loss, var_list=W[1:])
  init_op = tf.global_variables_initializer()
  sess = tf.InteractiveSession()
  sess.run(init_op, feed_dict=init_dict)
  
  for step in range(10):
    loss0 = loss.eval()
    sess.run(train_op)
    print("Step %d loss %.2f"%(step, loss0))
