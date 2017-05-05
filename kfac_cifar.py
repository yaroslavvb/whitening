import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, default='run', help='record to record test data, test to perform test, run to run training for longer')
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
parser.add_argument('--method', type=str, default="kfac", help='turn on KFAC')
parser.add_argument('--fixed_labels', type=int, default=0,
                    help='if true, fix synthetic labels to all 1s')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate to use')
parser.add_argument('--validate_every_n', type=int, default=10,
                    help='set to positive number to measure validation')
# lambda tuning graphs: https://wolfr.am/lojcyhYz
parser.add_argument('--Lambda', type=float, default=0.01,
                    help='lambda value')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

use_tikhonov=False

if args.mode == 'run':
  num_steps=10000
  LR=args.lr
  LAMBDA=args.Lambda
  use_fixed_labels = args.fixed_labels
else:
  num_steps = 10
#  LR=0.001
  LAMBDA=1e-1
  use_fixed_labels = True
  args.seed = 1

prefix="kfac_cifar"
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

  from keras.datasets import cifar10
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()

  X_train = X_train.astype(dtype)
  X_train = X_train.reshape((X_train.shape[0], -1))
  X_test = X_test.astype(dtype)
  X_test = X_test.reshape((X_test.shape[0], -1))
  X_train /= 255
  X_test /= 255

  train_images = X_train.T
  patches = train_images[:,:batch_size];
  test_patches = X_test.T
  test_patches = test_patches[:,:batch_size];

  input_dim = 3*32*32
  fs = [batch_size, input_dim, 1024, 1024, 1024, 196, 1024, 1024, 1024,
        input_dim]
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
  model.loss = u.L2(err) / (2 * batch_size)

  # create test error eval
  layer0 = init_var(test_patches, "X_test", is_global=False)
  layer = layer0
  for i in range(1, n+1):
    layer = nonlin(W[i] @ layer)
  verr = (layer - layer0)
  model.vloss = u.L2(verr) / (2 * batch_size)

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
      print("Initializing following:")
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

  return model

if __name__ == '__main__':
  np.random.seed(args.seed)
  tf.set_random_seed(args.seed)

  if args.mode == 'run':
    dsize = 100
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
  #  kfac.lr.set(LR)  # this is only used for adaptive_step
  kfac.Lambda.set(LAMBDA)

  with u.capture_vars() as opt_vars:
    if args.mode != 'run':
      opt = tf.train.AdamOptimizer(0.001)
    else:
      opt = tf.train.AdamOptimizer(LR)
      
    grads_and_vars = opt.compute_gradients(model.loss,
                                           var_list=model.trainable_vars)
    grad = IndexedGrad.from_grads_and_vars(grads_and_vars)
    grad_new = kfac.correct(grad)
    train_op = opt.apply_gradients(grad_new.to_grads_and_vars())
  [v.initializer.run() for v in opt_vars]
  
  losses = []
  u.record_time()

  start_time = time.time()
  vloss0 = 0
  
  outfn = 'data/%s_%f_%f.csv'%(prefix, args.lr, args.Lambda)

  start_time = time.time()
  for step in range(num_steps):
    if args.validate_every_n and step%args.validate_every_n == 0:
      loss0, vloss0 = sess.run([model.loss, model.vloss])
    else:
      loss0, = sess.run([model.loss])
    losses.append(loss0)

    elapsed = time.time()-start_time
    print("%d sec, step %d, loss %.2f, vloss %.2f" %(elapsed, step, loss0,
                                                     vloss0))

    # todo: factor out buffering into util
    with open(outfn, "a") as myfile:
      myfile.write('%d, %f, %f, %f\n'%(step, elapsed, loss0, vloss0))

    if args.method=='kfac':
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

