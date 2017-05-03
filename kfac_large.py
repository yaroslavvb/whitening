import argparse
import json
import os
import sys
import time
use_kfac = True

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, default='run', help='record to record test data, test to perform test, run to run training for longer')
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

LR=0.02  # TODO: two places to set lr
LAMBDA=1e-1
use_tikhonov=False

if args.mode == 'run':
  num_steps = 100
else:
  num_steps = 10
  
use_fixed_labels = True
hack_global_init_dict = {}

prefix="kfac_large"
script_fn = sys.argv[0].split('.', 1)[0]
assert prefix.startswith(script_fn)

# Test implementation of KFAC on MNIST
import load_MNIST

import util as u
import util
from util import t  # transpose

import kfac as kfac_lib
from kfac import Model
from kfac import Kfac
from kfac import IndexedGrad
import kfac

import sys
import tensorflow as tf
import numpy as np


rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# TODO: get rid of this
purely_linear = False  # convert sigmoids into linear nonlinearities
purely_relu = True     # convert sigmoids into ReLUs

regularized_svd = True # kfac_lib.regularized_svd # TODO: delete this


# TODO: get rid
def W_uniform(s1, s2): # uniform weight init from Ng UFLDL
  r = np.sqrt(6) / np.sqrt(s1 + s2 + 1)
  result = np.random.random(2*s2*s1)*2*r-r
  return result


def ng_init(rows, cols):
  # creates uniform initializer using Ng's formula
  # TODO: turn into TF
  r = np.sqrt(6) / np.sqrt(rows + cols + 1)
  result = np.random.random(rows*cols)*2*r-r
  return result.reshape((rows, cols))


def model_creator(batch_size, dtype=np.float32):
  """Create MNIST autoencoder model. Dataset is part of model."""

  global hack_global_init_dict
  
  model = Model()

  init_dict = {}
  global_vars = []
  local_vars = []
  
  # TODO: factor out to reuse between scripts
  # TODO: change feed_dict logic to reuse value provided to VarStruct
  # current situation makes reinitialization of global variable change
  # it's value, counterinituitive
  def init_var(val, name, is_global=False):
    """Helper to create variables with numpy or TF initial values."""
    print("Initializing %s with dtype %s"%(name, val.dtype))
    if isinstance(val, tf.Tensor):
      var = u.get_variable(name=name, initializer=val, reuse=is_global)
    else:
      val = np.array(val)
      assert u.is_numeric(val), "Non-numeric type."
      
      var_struct = u.get_var(name=name, initializer=val, reuse=is_global)
      holder = var_struct.val_
      init_dict[holder] = val
      var = var_struct.var

    if is_global:
      global_vars.append(var)
    else:
      local_vars.append(var)
      
    return var

  # TODO: get rid of purely_relu
  def nonlin(x):
    if purely_relu:
      return tf.nn.relu(x)
    elif purely_linear:
      return tf.identity(x)
    else:
      return tf.sigmoid(x)

  # TODO: rename into "nonlin_d"
  def d_nonlin(y):
    if purely_relu:
      return u.relu_mask(y)
    elif purely_linear:
      return 1
    else: 
      return y*(1-y)
    
  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte').astype(dtype)
  patches = train_images[:,:batch_size];
  fs = [batch_size, 28*28, 1024, 1024, 1024, 196, 1024, 1024, 1024, 28*28]
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  n = len(fs) - 2

  X = init_var(patches, "X", is_global=False)
  W = [None]*n
  W.insert(0, X)
  A = [None]*(n+2)
  A[1] = W[0]
  for i in range(1, n+1):
    init_val = ng_init(f(i), f(i-1)).astype(dtype)
    W[i] = init_var(init_val, "W_%d"%(i,), is_global=True)
    A[i+1] = nonlin(kfac_lib.matmul(W[i], A[i]))
    
  err = A[n+1] - A[1]

  # manually compute backprop to use for sanity checking
  B = [None]*(n+1)
  B2 = [None]*(n+1)
  B[n] = err*d_nonlin(A[n+1])
  _sampled_labels_live = tf.random_normal((f(n), f(-1)), dtype=dtype, seed=0)
  if use_fixed_labels:
    _sampled_labels_live = tf.ones(shape=(f(n), f(-1)), dtype=dtype)
    
  _sampled_labels = init_var(_sampled_labels_live, "to_be_deleted",
                             is_global=False)

  B2[n] = _sampled_labels*d_nonlin(A[n+1])
  for i in range(n-1, -1, -1):
    backprop = t(W[i+1]) @ B[i+1]
    B[i] = backprop*d_nonlin(A[i+1])
    backprop2 = t(W[i+1]) @ B2[i+1]
    B2[i] = backprop2*d_nonlin(A[i+1])

  cov_A = [None]*(n+1)    # covariance of activations[i]
  cov_B2 = [None]*(n+1)   # covariance of synthetic backprops[i]
  vars_svd_A = [None]*(n+1)
  vars_svd_B2 = [None]*(n+1)
  dW = [None]*(n+1)
  dW2 = [None]*(n+1)
  pre_dW = [None]*(n+1)   # preconditioned dW
  for i in range(1,n+1):
    if regularized_svd:
      cov_A[i] = init_var(A[i]@t(A[i])/batch_size+LAMBDA*u.Identity(f(i-1)), "cov_A%d"%(i,))
      cov_B2[i] = init_var(B2[i]@t(B2[i])/batch_size+LAMBDA*u.Identity(f(i)), "cov_B2%d"%(i,))
    else:
      cov_A[i] = init_var(A[i]@t(A[i])/batch_size, "cov_A%d"%(i,))
      cov_B2[i] = init_var(B2[i]@t(B2[i])/batch_size, "cov_B2%d"%(i,))
    vars_svd_A[i] = u.SvdWrapper(cov_A[i],"svd_A_%d"%(i,))
    vars_svd_B2[i] = u.SvdWrapper(cov_B2[i],"svd_B2_%d"%(i,))
    if use_tikhonov:
      whitened_A = u.regularized_inverse3(vars_svd_A[i],L=LAMBDA) @ A[i]
      whitened_B2 = u.regularized_inverse3(vars_svd_B2[i],L=LAMBDA) @ B[i]
    else:
      whitened_A = u.pseudo_inverse2(vars_svd_A[i]) @ A[i]
      whitened_B2 = u.pseudo_inverse2(vars_svd_B2[i]) @ B[i]
    
    dW[i] = (B[i] @ t(A[i]))/batch_size
    dW2[i] = B[i] @ t(A[i])
    pre_dW[i] = (whitened_B2 @ t(whitened_A))/batch_size

    #  model.extra['A'] = A
    #  model.extra['B'] = B
    #  model.extra['B2'] = B2
    #  model.extra['cov_A'] = cov_A
    #  model.extra['cov_B2'] = cov_B2
    #  model.extra['vars_svd_A'] = vars_svd_A
    #  model.extra['vars_svd_B2'] = vars_svd_B2
    #  model.extra['W'] = W
    #  model.extra['dW'] = dW
    #  model.extra['dW2'] = dW2
    #  model.extra['pre_dW'] = pre_dW
    
  model.loss = u.L2(err) / (2 * batch_size)
  sampled_labels_live = A[n+1] + tf.random_normal((f(n), f(-1)),
                                                  dtype=dtype, seed=0)
  if use_fixed_labels:
    sampled_labels_live = A[n+1]+tf.ones(shape=(f(n), f(-1)), dtype=dtype)
  sampled_labels = init_var(sampled_labels_live, "sampled_labels", is_global=False)
  err2 = A[n+1] - sampled_labels
  model.loss2 = u.L2(err2) / (2 * batch_size)
  model.global_vars = global_vars
  model.local_vars = local_vars
  model.trainable_vars = W[1:]

  def advance_batch():
    sess = tf.get_default_session()
    # TODO: get rid of _sampled_labels
    sess.run([sampled_labels.initializer, _sampled_labels.initializer])
  model.advance_batch = advance_batch

  # TODO: refactor this to take initial values out of Var struct
  #global_init_op = tf.group(*[v.initializer for v in global_vars])
  global_init_ops = [v.initializer for v in global_vars]
  global_init_op = tf.group(*[v.initializer for v in global_vars])
  
  global_init_query_op = [tf.logical_not(tf.is_variable_initialized(v))
                          for v in global_vars]
  def initialize_global_vars(verbose=False, reinitialize=False):
    """If reinitialize is false, will not reinitialize variables already
    initialized."""
    
    sess = tf.get_default_session()
    if not reinitialize:
      uninited = sess.run(global_init_query_op)
      # use numpy boolean indexing to select list of initializers to run
      to_initialize = list(np.asarray(global_init_ops)[uninited])
    else:
      to_initialize = global_init_ops
      
    if verbose:
      print("Initializing following global variables:")
      for v in to_initialize:
        print("   " + v.name)
        
    sess.run(to_initialize, feed_dict=init_dict)
  model.initialize_global_vars = initialize_global_vars

  local_init_op = tf.group(*[v.initializer for v in local_vars])
  def initialize_local_vars():
    sess = tf.get_default_session()
    sess.run(X.initializer, feed_dict=init_dict)  # A's depend on X
    sess.run(_sampled_labels.initializer, feed_dict=init_dict)
    sess.run(local_init_op, feed_dict=init_dict)
  model.initialize_local_vars = initialize_local_vars

  hack_global_init_dict = init_dict  # TODO: remove hack
  
  return model

if __name__ == '__main__':
  np.random.seed(0)
  tf.set_random_seed(0)

  if args.mode == 'run':
    dsize = 10000
  else:
    dsize = 1000
    
  sess = tf.InteractiveSession()
  model = model_creator(dsize) # TODO: share dataset between models?
  model.initialize_global_vars(verbose=True)
  model.initialize_local_vars()
  
  kfac = Kfac(model_creator, dsize)   # creates another copy of model, initializes


  kfac.model.initialize_global_vars(verbose=True)
  kfac.model.initialize_local_vars()
  kfac.reset()    # resets optimization variables (not model variables)
  kfac.lr.set(LR)
  kfac.Lambda.set(LAMBDA)

  with u.capture_vars() as opt_vars:
    if use_kfac:
      opt = tf.train.AdamOptimizer(0.001)
    else:
      opt = tf.train.AdamOptimizer()
      
    grads_and_vars = opt.compute_gradients(model.loss,
                                           var_list=model.trainable_vars)
    grad = IndexedGrad.from_grads_and_vars(grads_and_vars)
    grad_new = kfac.correct(grad)
    train_op = opt.apply_gradients(grad_new.to_grads_and_vars())
    
      
  [v.initializer.run() for v in opt_vars]
  
  losses = []
  u.record_time()

  start_time = time.time()
  for step in range(num_steps):
    loss0 = model.loss.eval()
    losses.append(loss0)
    elapsed = time.time()-start_time
    print("%d sec, step %d, loss %.2f" %(elapsed, step, loss0))

    if use_kfac:
      kfac.model.advance_batch()
      kfac.update_stats()

    model.advance_batch()
    grad.update()
    grad_new.update()
    train_op.run()
    
    u.record_time()

  u.summarize_time()
  
  losses_fn = '%s_losses_test.csv' %(prefix,)
  if args.mode == 'record':
    u.dump_with_prompt(losses, losses_fn)

  elif args.mode == 'test':
    targets = np.loadtxt("data/"+losses_fn, delimiter=",")
    u.check_equal(losses, targets, rtol=1e-2)
    u.summarize_difference(losses, targets)

