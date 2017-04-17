use_preconditioner = True
adaptive_step = False
drop_l2 = True
drop_sparsity = True
drop_reconstruction = False
do_single_core = False
use_gpu = False

import sys
#whitening_mode = int(sys.argv[1])
whitening_mode=3
whiten_every_n_steps = 1

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
if use_gpu:
  os.environ['CUDA_VISIBLE_DEVICES']='0'
else:
  os.environ['CUDA_VISIBLE_DEVICES']=''

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
  result = np.random.random(2*s2*s1)*2*r-r
  #  u.dump(result, "W2.csv")
  return result


if __name__=='__main__':
  np.random.seed(0)
  tf.set_random_seed(0)
  dtype = np.float32
  u.default_dtype = dtype
  
  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  dsize = 5000
  patches = train_images[:,:dsize];
  fs = [dsize, 28*28, 196, 28*28]

  #  u.dump(patches, "mnist.csv")
  #  sys.exit()

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

  lr = init_var(0.2, "lr")
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

  # A[0] is just for shape checks, assert fail on run
  with tf.control_dependencies([tf.assert_equal(1, 0, message="too huge")]):
    A[0] = u.Identity(dsize, dtype=dtype)
  A[1] = W[0]
  for i in range(1, n+1):
    A[i+1] = sigmoid(W[i] @ A[i])
    

  # reconstruction error and sparsity error
  err = (A[3] - A[1])
  rho_hat = tf.reduce_sum(A[2], axis=1, keep_dims=True)/dsize

  # B[i] = backprops needed to compute gradient of W[i]
  # B2[i] = synthetic backprops for natural gradient
  B = [None]*(n+1)
  B2 = [None]*(n+1)
  B[n] = err*d_sigmoid(A[n+1])
  B2[n] = tf.random_normal((f(n), f(-1)),dtype=dtype,seed=0)*d_sigmoid(A[n+1])
  for i in range(n-1, -1, -1):
    backprop = t(W[i+1]) @ B[i+1]
    backprop2 = t(W[i+1]) @ B2[i+1]
    if i == 1 and not drop_sparsity:
      backprop += beta*d_kl(rho, rho_hat)
      backprop2 += beta*d_kl(rho, rho_hat)
    B[i] = backprop*d_sigmoid(A[i+1])
    B2[i] = backprop2*d_sigmoid(A[i+1])

  # dW[i] = gradient of W[i]
  dW = [None]*(n+1)
  dW2 = [None]*(n+1)
  Acov = [None]*(n+1)
  Bcov = [None]*(n+1)    # empirical covariances
  Bcov2 = [None]*(n+1)   # natural gradient sampled covariances
  whitenA = [None]*(n+1)
  whitenB = [None]*(n+1)
  for i in range(n+1):
    whitenA[i] = tf.Variable(u.Identity(f(i-1)))
    whitenB[i] = tf.Variable(u.Identity(f(i)))
    dW[i] = (B[i]) @ t(A[i])/dsize
    if use_preconditioner:
      dW2[i] = (whitenB[i] @ B[i]) @ t(whitenA[i] @ A[i])/dsize
    else:
      dW2[i] = (B[i]) @ t(A[i])/dsize
    Acov[i] = A[i]@t(A[i])/dsize
    Bcov[i] = B[i]@t(B[i])/dsize
    Bcov2[i] = B2[i]@t(B2[i])/dsize

  # Cost function
  reconstruction = u.L2(err) / (2 * dsize)
  sparsity = beta * tf.reduce_sum(kl(rho, rho_hat))
  L2 = (lambda_ / 2) * (u.L2(W[1]) + u.L2(W[1]))

  cost = 0
  if not drop_reconstruction:
    cost = cost + reconstruction
  if not drop_l2:
    cost = cost + L2
  if not drop_sparsity:
    cost = cost + sparsity

  grad = u.flatten(dW[1:])    # true gradient
  grad2 = u.flatten(dW2[1:])  # preconditioned gradient
  copy_op = Wf_copy.assign(Wf-lr*grad2)
  with tf.control_dependencies([copy_op]):
    train_op = tf.group(Wf.assign(Wf_copy)) # to make it an op

  if do_single_core:
    sess = tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1,intra_op_parallelism_threads=1))
  else:
    sess = tf.InteractiveSession()

  #  step_len = init_var(tf.constant(0.1), "step_len", False)
  #  step_len_assign = step_len.assign(step_len0)
  step_len0 = tf.placeholder(dtype, shape=())
  
  Wf2 = init_var(W0f, "Wf2")
  Wf_save_op = Wf2.assign(Wf)
  Wf_restore_op = Wf.assign(Wf2)
  grad_copy = init_var(W0f, "grad_copy")
  grad2_copy = init_var(W0f, "grad2_copy")
  direction = init_var(W0f, "direction")
  
  grad_save_op = grad_copy.assign(grad)
  grad2_save_op = grad2_copy.assign(grad2)
  grad_copy_norm_op = tf.reduce_sum(tf.square(grad_copy))
  
  grad2_dot_grad_op = tf.reduce_sum(grad2_copy*grad_copy)
  
  Wf_step_op = Wf.assign(Wf2 - step_len0*grad_copy)
  lr_p = tf.placeholder(lr.dtype, lr.shape)
  lr_set = lr.assign(lr_p)

  def save_wf(): sess.run(Wf_save_op)
  def restore_wf(): sess.run(Wf_restore_op)
  def save_grad(): sess.run(grad_save_op)
  def save_grad2(): sess.run(grad2_save_op)
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
  costs = []
  ratios = []
  # adaptive line search parameters
  alpha=0.3   # acceptable fraction of predicted decrease
  beta=0.8    # how much to shrink when violation
  growth_rate = 1.05  # how much to grow when too conservative

  if whitening_mode>0:
    sess.run(whitenA[1].assign(u.pseudo_inverse_sqrt(Acov[1],eps=1e-20)))
  #  sess.run(whitenB[1].assign(u.pseudo_inverse_sqrt(Acov[1],eps=1e-20)))
  
  for i in range(5):
    # save Wf and grad into Wf2 and grad_copy
    save_wf()
    save_grad()  # => grad_copy
    save_grad2()  # => grad_copy
    cost0 = cost.eval()
    train_op.run()
    lr0 = lr.eval()
    cost1 = cost.eval()
    #    cost1, _ = sess.run([cost, train_op])
    #    target_delta = -alpha*lr0*grad_copy_norm_op.eval()
    target_delta = -alpha*lr0*grad2_dot_grad_op.eval()
    expected_delta = -lr0*grad2_dot_grad_op.eval()

    actual_delta = cost1 - cost0
    actual_slope = actual_delta/lr0
    #    expected_slope = -grad_copy_norm_op.eval()
    expected_slope = -grad2_dot_grad_op.eval()

    # ratio of best possible slope to actual slope
    # don't divide by actual slope because that can be 0
    slope_ratio = abs(actual_slope)/abs(expected_slope)
    costs.append(cost0)
    step_lengths.append(lr0)
    ratios.append(slope_ratio)

    if i%whiten_every_n_steps==0:
      pass
      # each is about 200 ms
      if whitening_mode>1:
        sess.run(whitenA[2].assign(u.pseudo_inverse_sqrt(Acov[2],eps=1e-20)))
      if whitening_mode>2:
        sess.run(whitenB[2].assign(u.pseudo_inverse_sqrt(Bcov2[2],eps=1e-20)))
      if whitening_mode>3:
      # Get NaN's if I whiten B[1] as well
        sess.run(whitenB[1].assign(u.pseudo_inverse_sqrt(Bcov2[1],eps=1e-20)))

    print("Step %d cost %.2f, expected decrease %.3f, actual decrease, %.3f ratio %.2f"%(i, cost0, expected_delta, actual_delta, slope_ratio))
    if i%10 == 0:
      pass
      #      print("Cost %.2f, expected decrease %.3f, actual decrease, %.3f ratio %.2f"%(cost0, expected_delta, actual_delta, slope_ratio))
      #      for layer_num in range(1, n+1):
      #        u.dump(Acov[layer_num], "Acov-%d-%d.csv"%(layer_num, i,))
      #        u.dump(Bcov[layer_num], "Bcov-%d-%d.csv"%(layer_num, i,))
      
      # if len(costs)>6 and costs[-5]-costs[-1] < 0.0001:
      #   print("Converged in %d to %.2f "%(i, cost0))
      #   break<

    
    # don't shrink learning rate once results are very close to minimum
    if slope_ratio < alpha and abs(target_delta)>1e-6 and adaptive_step:
      print("%.2f %.2f %.2f"%(cost0, cost1, slope_ratio))
      print("Slope optimality %.2f, shrinking learning rate to %.2f"%(slope_ratio, lr0*beta,))
      sess.run(lr_set, feed_dict={lr_p: lr0*beta})
    else:
      # see if our learning rate got too conservative, and increase it
      # 99 was ideal for gradient
#      if i>0 and i%50 == 0 and slope_ratio>0.99:
      if i>0 and i%50 == 0 and slope_ratio>0.90 and adaptive_step:
        print("%.2f %.2f %.2f"%(cost0, cost1, slope_ratio))
        print("Growing learning rate to %.2f"%(lr0*growth_rate))
        sess.run(lr_set, feed_dict={lr_p: lr0*growth_rate})

    if do_images and i>0 and i%100==0:
      Wf_ = sess.run("Wf_var/read:0")
      W1_ = u.unflatten_np(Wf_, fs[1:])[0]
      display_network.display_network(W1_.T, filename="pics/weights-%03d.png"%(frame_count,))
      frame_count+=1
      old_cost = cost0
      old_i = i


    u.record_time()

    #  u.dump(costs, "mac1.csv")
  targets = np.loadtxt("data/mac1.csv", delimiter=",")
  u.check_equal(costs[:5], targets[:5])
  u.summarize_time()
