"""Do line searches, dump csvs"""

import load_MNIST
import sparse_autoencoder
import scipy.optimize
import softmax
import numpy as np
import display_network
from IPython.display import Image
import scipy.io # for loadmat
import matplotlib # for matplotlib.cm.gray
from matplotlib.pyplot import imshow
import math
import time

import os, sys
os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
import util as u
from util import t  # transpose

from util import t  # transpose
from util import c2v
from util import v2c
from util import v2c_np
from util import v2r
from util import kr  # kronecker
from util import Kmat # commutation matrix

def W_uniform(s1, s2):
  # sample two s1,s2 matrices 
  r = np.sqrt(6) / np.sqrt(s1 + s2 + 1)
  return np.random.random(2*s2*s1)*2*r-r


if __name__=='__main__':
  np.random.seed(0)
  tf.set_random_seed(0)
  dtype = np.float32
  
  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  dsize = 10000
  patches = train_images[:,:dsize];
  fs = [dsize, 28*28, 196, 28*28]

  fs=fs
  X0=patches
  lambda_=3e-3
  rho=0.1
  beta=3
  W0f=None
  
  if not W0f:
    W0f = W_uniform(fs[2],fs[3])
  rho = tf.constant(rho, dtype=dtype)

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = f(-1)
  n = len(fs) - 2

  init_dict = {}
  def init_var(val, name, trainable=False):
    val = np.array(val)
    holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
    var = tf.Variable(holder, name=name+"_var", trainable=trainable)
    init_dict[holder] = val
    return var

  lr = init_var(0.1, "lr")
  Wf = init_var(W0f, "Wf", True)
  Wf_copy = init_var(W0f, "Wf_copy")
  W = u.unflatten(Wf, fs[1:])
  X = init_var(X0, "X")
  W.insert(0, X)

  def sigmoid(x):
    return tf.sigmoid(x)
  def d_sigmoid(y):
    return y*(1-y)
  def kl(x, y):
    return x * tf.log(x / y) + (1 - x) * tf.log((1 - x) / (1 - y))
  def d_kl(x, y):
    return (1-x)/(1-y) - x/y
  
  # A[i] = activations needed to compute gradient of W[i]
  # A[n+1] = network output
  A = [None]*(n+2)
  A[0] = u.Identity(dsize, dtype=dtype)
  A[1] = W[0]
  for i in range(1, n+1):
    A[i+1] = sigmoid(W[i] @ A[i])
    

  # reconstruction error and sparsity error
  err = (A[3] - A[1])
  rho_hat = tf.reduce_sum(A[2], axis=1, keep_dims=True)/dsize

  # B[i] = backprops needed to compute gradient of W[i]
  B = [None]*(n+1)
  B[n] = err*d_sigmoid(A[n+1])
  for i in range(n-1, -1, -1):
    backprop = t(W[i+1]) @ B[i+1]
    if i == 1:
      backprop += beta*d_kl(rho, rho_hat)
    B[i] = backprop*d_sigmoid(A[i+1])

  # dW[i] = gradient of W[i]
  dW = [None]*(n+1)
  for i in range(n+1):
    dW[i] = (B[i] @ t(A[i]))/dsize

  # Cost function
  reconstruction = u.L2(err) / (2 * dsize)
  sparsity = beta * tf.reduce_sum(kl(rho, rho_hat))
  L2 = (lambda_ / 2) * (u.L2(W[1]) + u.L2(W[1]))
  cost = reconstruction + sparsity + L2

  grad = u.flatten(dW[1:])
  copy_op = Wf_copy.assign(Wf-lr*grad)
  with tf.control_dependencies([copy_op]):
    train_op = Wf.assign(Wf_copy)

  sess = tf.InteractiveSession()

  #  step_len = init_var(tf.constant(0.1), "step_len", False)
  #  step_len_assign = step_len.assign(step_len0)
  step_len0 = tf.placeholder(dtype, shape=())
  
  Wf2 = init_var(W0f, "Wf2")
  Wf_save_op = Wf2.assign(Wf)
  Wf_restore_op = Wf.assign(Wf2)
  grad2 = init_var(W0f, "grad2")
  grad_save_op = grad2.assign(grad)
  Wf_step_op = Wf.assign(Wf2 - step_len0*grad2)
  lr_p = tf.placeholder(lr.dtype, lr.shape)
  lr_set = lr.assign(lr_p)

  def save_wf(): sess.run(Wf_save_op)
  def restore_wf(): sess.run(Wf_restore_op)
  def save_grad(): sess.run(grad_save_op)
  def step_wf(step):
    #    sess.run(step_len_assign, feed_dict={step_len0: step})
    sess.run(Wf_step_op, feed_dict={step_len0: step}) 
  
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

  
  print("Running training.")
  do_images = True
  u.reset_time()
  old_cost = sess.run(cost)
  old_i = 0
  frame_count = 0

  step_lengths = []
  losses = []
  do_bt = True  # do backtracking line-search
  for i in range(1000):
    cost0, _ = sess.run([cost, train_op])
    losses.append(cost0)
    if i>0 and i%50==0:
      print(cost0)
      if step_lengths:
        print("      ", step_lengths[-1])
    if do_bt and i%10>=0 and i%10<=2:
      #      save_wf()
      #      save_grad()
      
      # print(cost0)
      # # do line search
      # min_lr = -0.1
      # max_lr = 200.
      # save_wf()
      # save_grad()
      # num_steps = 20
      # line_search_costs = []
      # for j in range(num_steps):
      #   actual_step = min_lr + j*(max_lr-min_lr)/num_steps
      #   step_wf(actual_step)
      #   line_search_costs.append([actual_step, cost.eval()-cost0])
      # u.dump(line_search_costs, "linesearch-%d.csv"%(i,))
      # restore_wf()

      save_wf()
      save_grad()
      # do backtracking line search
      grad2_ = grad2.eval()
      alpha=0.3
      beta=0.8
      cost0 = cost.eval()

      def f(t):  # returns cost difference along direction
        step_wf(t)
        return cost.eval() - cost0
      t = 1
      bt_costs = []
      while True:
        target_delta = -alpha*t*np.square(grad2_).sum()
        actual_delta = f(t)
        bt_costs.append(actual_delta)
        #        print("Target delta %s, actual delta %s"%(target_delta,
        #                                                  actual_delta))
        if actual_delta > target_delta:
          t = t*beta
        else:
          break
      print("Setting step "+str(t))
      sess.run(lr_set, feed_dict={lr_p: t})
      step_lengths.append(t)
      restore_wf()
    u.record_time()

#  u.dump(losses, "losses_regular.csv")
      
#  u.dump(losses, "losses_bt.csv")
#  u.dump(step_lengths, "step_lengths_bt.csv")

# BT search every 10 steps
#  u.dump(losses, "losses_bt2.csv")
#  u.dump(step_lengths, "step_lengths_bt2.csv")
#  u.dump(losses, "losses_bt3.csv")
#  u.dump(step_lengths, "step_lengths_bt3.csv")
#  u.summarize_time()

  # double learning rate
  #  u.dump(losses, "losses_regular2.csv")

  # do 3 backtracking line searches every 10 steps
  u.dump(losses, "losses_bt4.csv")
  u.dump(step_lengths, "step_lengths_bt4.csv")
