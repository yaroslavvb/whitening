# KFAC implementation and test on MNIST
# adam iteration time: 8ms 
# iteration times in ms on Pascal 1080 TI: min: 7.39, median: 8.02
prefix = "gd"
prefix = "adam"  # default gradient lr=1e-3
prefix = "adam2" # try lr=0.01 (also tried 1.0, 0.1, they diverge)
prefix = "adam_bn" # relu with with batch norm on first matmul, lr 0.01 sucks, do 0.001 instead

prefix = "adam_no_bn" # same as previous, but batch norm removed
prefix = "temp"

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
use_gpu = True

import tensorflow as tf
import util
import util as u
from util import t  # transpose

import load_MNIST

use_batch_norm = False
num_steps = 20000
whitening_mode = 0

whiten_every_n_steps = 1
report_frequency = 1

if use_gpu:
  os.environ['CUDA_VISIBLE_DEVICES']='0'
else:
  os.environ['CUDA_VISIBLE_DEVICES']=''

def W_uniform(s1, s2):
  r = np.sqrt(6) / np.sqrt(s1 + s2 + 1)
  result = np.random.random(2*s2*s1)*2*r-r
  return result

"""Every matmul has an associated parameter vector.

update_grad
update_pre_grad
update_pre_grad2.

W, dW, dW_pre, dW_pre2



"""

def kfac_optimizer(model_creator):
  stats_batch_size = 10000
  main_batch_size = 10000
  
  stats_model, loss, labels = model_creator(stats_batch_size)
  # replace labels_node with synthetic labels

  main_model, _, _ = model_creator(main_batch_size)

  opt = tf.GradientDescentOptimizer(0.2)
  grads_and_vars = opt.compute_gradients(loss)

  trainable_vars = tf.trainable_variables()

  # create SVD and preconditioning variables for matmul vars
  for var in trainable_vars:
    if var not in matmul_registry:
      continue
    dW = u.extract_grad(grads_and_vars, var)
    A[var] = get_activations(var)
    B[var] = get_backprops(var)
    B2[var] = get_backprops2(var) # get backprops with synthetic labels
    dW[var] = B[var]@t(A[var]) # todo: sort out dsize division
    cov_A[var] = init_var(A[var]@t(A[var])/dsize, "cov_A_%s"%(var.name,))
    cov_B2[var] = init_var(B2[var]@t(B2[var])/dsize, "cov_B2_%s"%(var.name,))

    vars_svd_A[var] = SvdWrapper(cov_A[var],"svd_A_%d"%(var.name,))
    vars_svd_B2[var] = SvdWrapper(cov_B2[var],"svd_B2_%d"%(var.name,))
    whitened_A = u.pseudo_inverse2(vars_svd_A[var]) @ A[var]
    whitened_B2 = u.pseudo_inverse2(vars_svd_B2[var]) @ B[var]
    whitened_A_stable = u.pseudo_inverse_sqrt2(vars_svd_A[var]) @ A[var]
    whitened_B2_stable = u.pseudo_inverse_sqrt2(vars_svd_B2[var]) @ B[var]
    
    pre_dW[var] = (whitened_B2 @ t(whitened_A))/dsize
    pre_dW_stable[var] = (whitened_B2_stable @ t(whitened_A_stable))/dsize
    dW[var] = (B[var] @ t(A[var]))/dsize
      
  # create update params ops
  
  # new_grads_and_vars = []
  # for grad, var in grads_and_vars:
  #   if var in kfac_registry:
  #     pre_A, pre_B = kfac_registry[var]
  #     new_grad_live = pre_B @ grad @ t(pre_A)
  #     new_grads_and_vars.append((new_grad, var))
  #     print("Preconditioning %s"%(var.name))
  #   else:
  #     new_grads_and_vars.append((grad, var))
  # train_op = opt.apply_gradients(new_grads_and_vars)

  # Each variable has an associated gradient, pre_gradient, variable save op
  def update_grad():
    ops = [grad_update_ops[var] for var in trainable_vars]
    sess.run(ops)

  def update_pre_grad():
    ops = [pre_grad_update_ops[var] for var in trainable_vars]
    sess.run(ops)
    
  def update_pre_grad2():
    ops = [pre_grad2_update_ops[var] for var in trainable_vars]
    sess.run(ops)
    
  def save_params():
    ops = [var_save_ops[var] for var in trainable_vars]
    sess.run(ops)

    
  for step in range(num_steps):
    update_covariances()
    if step%whitened_every_n_steps == 0:
      update_svds()

    update_grad()
    update_pre_grad()   # perf todo: update one of these
    update_pre_grad2()  # stable alternative
    
    lr0, loss0 = sess.run([lr, loss])
    save_params()

    # when grad norm<1, Fisher is unstable, switch to Sqrt(Fisher)
    # TODO: switch to per-matrix normalization
    stabilized_mode = grad_norm.eval()<1

    if stabilized_mode:
      update_params2()
    else:
      update_params()

    loss1 = loss.eval()
    advance_batch()

    # line search stuff
    target_slope = (-pre_grad_dot_grad.eval() if stabilized_mode else
                    -pre_grad_stable_dot_grad.eval())
    target_delta = lr0*target_slope
    actual_delta = loss1 - loss0
    actual_slope = actual_delta/lr0
    slope_ratio = actual_slope/target_slope  # between 0 and 1.01 

    losses.append(loss0)
    step_lengths.append(lr0)
    ratios.append(slope_ratio)

    if step % report_frequency == 0:
      print("Step %d loss %.2f, target decrease %.3f, actual decrease, %.3f ratio %.2f grad norm: %.2f pregrad norm: %.2f"%(step, loss0, target_delta, actual_delta, slope_ratio, grad_norm.eval(), pre_grad_norm.eval()))
    
    u.record_time()


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

  def nonlin(x):
    return tf.nn.relu(x)
  #    return tf.sigmoid(x)

  A = [None]*(n+2)
  with tf.control_dependencies([tf.assert_equal(1, 0, message="too huge")]):
    A[0] = u.Identity(dsize, dtype=dtype)
  A[1] = W[0]
  for i in range(1, n+1):
    A[i+1] = nonlin(W[i] @ A[i])
    # add batch norm to first layer
    if use_batch_norm and i==1 and not prefix=='adam_no_bn':
      A[i+1] = tf.contrib.layers.batch_norm(A[i+1], 
                                            center=True, scale=True, 
                                            is_training=True)
    
  err = (A[3] - A[1])
  loss = u.L2(err) / (2 * dsize)

  #  opt = tf.train.GradientDescentOptimizer(0.2)
  opt = tf.train.AdamOptimizer(learning_rate=0.001)
  grads_and_vars = opt.compute_gradients(loss, var_list=W[1:])
  assert grads_and_vars[0][1] == W[1]
  train_op = opt.apply_gradients(grads_and_vars)

  init_op = tf.global_variables_initializer()
  sess = tf.InteractiveSession()
  sess.run(init_op, feed_dict=init_dict)

#   vars_svd_A = [None]*(n+1)
#   vars_svd_B2 = [None]*(n+1)
#   for i in range(1,n+1):
#     cov_A[i] = init_var(A[i]@t(A[i])/dsize, "cov_A%d"%(i,))
# #    cov_B2[i] = init_var(B2[i]@t(B2[i])/dsize, "cov_B2%d"%(i,))
#     vars_svd_A[i] = SvdWrapper(cov_A[i],"svd_A_%d"%(i,))
# #    vars_svd_B2[i] = SvdWrapper(cov_B2[i],"svd_B2_%d"%(i,))
#     whitened_A = u.pseudo_inverse2(vars_svd_A[i]) @ A[i]
# #    whitened_B2 = u.pseudo_inverse2(vars_svd_B2[i]) @ B[i]
#     whitened_A_stable = u.pseudo_inverse_sqrt2(vars_svd_A[i]) @ A[i]
# #    whitened_B2_stable = u.pseudo_inverse_sqrt2(vars_svd_B2[i]) @ B[i]
#     pre_dW[i] = (whitened_B2 @ t(whitened_A))/dsize
#     pre_dW_stable[i] = (whitened_B2_stable @ t(whitened_A_stable))/dsize
#     dW[i] = (B[i] @ t(A[i]))/dsize


  
  losses = []
  u.record_time()
  for step in range(num_steps):
    loss0 = loss.eval()
    losses.append(loss0)
    sess.run(train_op)
    if step % report_frequency == 0:
      print("Step %d loss %.2f"%(step, loss0))
    u.record_time()

  u.summarize_time()


  u.dump(losses, "%s_losses_%d.csv"%(prefix, whitening_mode,))
