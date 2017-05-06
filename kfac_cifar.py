#!/usr/bin/env python
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
parser.add_argument('-L', '--Lambda', type=float, default=0.01,
                    help='lambda value')
parser.add_argument('-r', '--run', type=str, default='default',
                    help='name of experiment run')
parser.add_argument('-n', '--num_steps', type=int, default=1000000,
                    help='number of steps')
parser.add_argument('--dataset', type=str, default="cifar",
                    help='which dataset to use')
# todo: split between optimizer batch size and stats batch size
parser.add_argument('-b', '--batch_size', type=int, default=10000,
                    help='batch size')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

use_tikhonov=False

release_name='kfac_cifar'  # release name fixes a specific test set
release_test_fn = 'data/'+release_name+'_losses_test.csv'

if args.mode == 'test':
  args.num_steps = 10
  args.dataset = 'cifar'
#  LR=0.001
  args.Lambda=1e-1
  args.fixed_labels = True
  args.seed = 1

prefix=args.run

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


rundir = u.setup_experiment_run_directory(args.run,
                                          safe_mode=(args.mode=='run'))
with open(rundir+'/args.txt', 'w') as f:
  f.write(json.dumps(vars(args), indent=4, separators=(',',':')))
  f.write('\n')

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


if args.dataset == 'cifar':
  # load data globally once
  from keras.datasets import cifar10
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()
  X_train = X_train.astype(np.float32)
  X_train = X_train.reshape((X_train.shape[0], -1))
  X_test = X_test.astype(np.float32)
  X_test = X_test.reshape((X_test.shape[0], -1))
  X_train /= 255
  X_test /= 255
elif args.dataset == 'mnist':
  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte').astype(np.float32)
  test_patches = train_images[:,-args.batch_size:]
  X_train = train_images[:,:args.batch_size].T
  X_test = test_patches.T
  
  
# todo: rename to better names
train_images = X_train.T
test_images = X_test.T

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

  patches = train_images[:,:args.batch_size];
  test_patches = test_images[:,:args.batch_size];

  if args.dataset == 'cifar':
    input_dim = 3*32*32
    fs = [args.batch_size, input_dim, 1024, 1024, 1024, 196, 1024, 1024, 1024,
          input_dim]
  elif args.dataset == 'mnist':
    input_dim = 28*28
    fs = [args.batch_size, input_dim, 1024, 1024, 1024, 196, 1024, 1024, 1024,
          input_dim]
  else:
    assert False
    
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  n = len(fs) - 2

  X = init_var(patches, "X", is_global=True)
  W = [None]*n
  W.insert(0, X)
  A = [None]*(n+2)
  A[1] = W[0]
  for i in range(1, n+1):
    init_val = ng_init(f(i), f(i-1)).astype(dtype)
    W[i] = init_var(init_val, "W_%d"%(i,), is_global=True)
    A[i+1] = nonlin(kfac_lib.matmul(W[i], A[i]))
  err = A[n+1] - A[1]
  model.loss = u.L2(err) / (2 * args.batch_size)

  # create test error eval
  layer0 = init_var(test_patches, "X_test", is_global=True)
  layer = layer0
  for i in range(1, n+1):
    layer = nonlin(W[i] @ layer)
  verr = (layer - layer0)
  model.vloss = u.L2(verr) / (2 * args.batch_size)

  # manually compute backprop to use for sanity checking
  B = [None]*(n+1)
  B2 = [None]*(n+1)
  B[n] = err*d_nonlin(A[n+1])
  _sampled_labels_live = tf.random_normal((f(n), f(-1)), dtype=dtype, seed=0)
  if args.fixed_labels:
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
      cov_A[i] = init_var(A[i]@t(A[i])/args.batch_size+args.Lambda*u.Identity(f(i-1)), "cov_A%d"%(i,))
      cov_B2[i] = init_var(B2[i]@t(B2[i])/args.batch_size+args.Lambda*u.Identity(f(i)), "cov_B2%d"%(i,))
    else:
      cov_A[i] = init_var(A[i]@t(A[i])/args.batch_size, "cov_A%d"%(i,))
      cov_B2[i] = init_var(B2[i]@t(B2[i])/args.batch_size, "cov_B2%d"%(i,))
    vars_svd_A[i] = u.SvdWrapper(cov_A[i],"svd_A_%d"%(i,))
    vars_svd_B2[i] = u.SvdWrapper(cov_B2[i],"svd_B2_%d"%(i,))
    if use_tikhonov:
      whitened_A = u.regularized_inverse3(vars_svd_A[i],L=args.Lambda) @ A[i]
      whitened_B2 = u.regularized_inverse3(vars_svd_B2[i],L=args.Lambda) @ B[i]
    else:
      whitened_A = u.pseudo_inverse2(vars_svd_A[i]) @ A[i]
      whitened_B2 = u.pseudo_inverse2(vars_svd_B2[i]) @ B[i]
    
    dW[i] = (B[i] @ t(A[i]))/args.batch_size
    dW2[i] = B[i] @ t(A[i])
    pre_dW[i] = (whitened_B2 @ t(whitened_A))/args.batch_size

    
  sampled_labels_live = A[n+1] + tf.random_normal((f(n), f(-1)),
                                                  dtype=dtype, seed=0)
  if args.fixed_labels:
    sampled_labels_live = A[n+1]+tf.ones(shape=(f(n), f(-1)), dtype=dtype)
  sampled_labels = init_var(sampled_labels_live, "sampled_labels", is_global=False)
  err2 = A[n+1] - sampled_labels
  model.loss2 = u.L2(err2) / (2 * args.batch_size)
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

    with u.timeit("global init run"):
      sess.run(to_initialize, feed_dict=init_dict)
  model.initialize_global_vars = initialize_global_vars

  local_init_op = tf.group(*[v.initializer for v in local_vars])
  print("Local vars:")
  for v in local_vars:
    print(v.name)
  def initialize_local_vars():
    sess = tf.get_default_session()
#    with u.timeit("x_initializer"):
      # todo: remove this initializer
      #      sess.run(X.initializer, feed_dict=init_dict)  # A's depend on X
    sess.run(_sampled_labels.initializer, feed_dict=init_dict)
    with u.timeit("local_init_op"):
      #sess.run(local_init_op, feed_dict=init_dict)
      sess.run(local_init_op)
  model.initialize_local_vars = initialize_local_vars

  return model

if __name__ == '__main__':
  np.random.seed(args.seed)
  tf.set_random_seed(args.seed)
  
  if args.mode == 'test':
    args.batch_size = 100

  with u.timeit("session"):
    gpu_options = tf.GPUOptions(allow_growth=False)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

  print("Graphdef %.2f MB" %(len(str(tf.get_default_graph().as_graph_def()))/1000000.))
  with u.timeit("model initialize"):
    with u.timeit("model_creator"):
      model = model_creator(args.batch_size) # TODO: share dataset between models?
    model.initialize_global_vars(verbose=True)
    model.initialize_local_vars()

  print("Graphdef %.2f MB" %(len(str(tf.get_default_graph().as_graph_def()))/1000000.))
  with u.timeit("kfac initialize"):
    with u.timeit("kfac()"):
      kfac = Kfac(model_creator, args.batch_size)   # creates another copy of model, initializes

    with u.timeit("kfac.initialize_global_vars()"):
      kfac.model.initialize_global_vars(verbose=False)
    with u.timeit("kfac.initialize_local_vars()"):
      kfac.model.initialize_local_vars()
    with u.timeit("kfac.reset()"):
      kfac.reset()    # resets optimization variables (not model variables)
    #  kfac.lr.set(LR)  # this is only used for adaptive_step
    with u.timeit("kfac.lambda()"):
      kfac.Lambda.set(args.Lambda)

  print("Graphdef %.2f MB" %(len(str(tf.get_default_graph().as_graph_def()))/1000000.))
  with u.capture_vars() as opt_vars:
    if args.mode != 'run':
      opt = tf.train.AdamOptimizer(0.001)
    else:
      opt = tf.train.AdamOptimizer(args.lr)
      
    grads_and_vars = opt.compute_gradients(model.loss,
                                           var_list=model.trainable_vars)
    grad = IndexedGrad.from_grads_and_vars(grads_and_vars)
    grad_new = kfac.correct(grad)
    train_op = opt.apply_gradients(grad_new.to_grads_and_vars())
  with u.timeit("adam initialize"):
    sess.run([v.initializer for v in opt_vars])
  
  losses = []
  u.record_time()

  start_time = time.time()
  vloss0 = 0

  # todo, unify the two data outputs
  outfn = 'data/%s_%f_%f.csv'%(prefix, args.lr, args.Lambda)

  start_time = time.time()
  sw = tf.summary.FileWriter('runs/'+prefix, sess.graph)
  
  writer = u.BufferedWriter(outfn, 60)
  for step in range(args.num_steps):
    
    if args.validate_every_n and step%args.validate_every_n == 0:
      loss0, vloss0 = sess.run([model.loss, model.vloss])
      sw.flush()
    else:
      loss0, = sess.run([model.loss])
    losses.append(loss0)

    summary = tf.Summary()
    summary.value.add(tag="loss", simple_value=float(loss0))
    summary.value.add(tag="vloss/1", simple_value=float(vloss0))
    summary.value.add(tag="vloss/2", simple_value=float(vloss0))
    sw.add_summary(summary, step)
    
    elapsed = time.time()-start_time
    print("%d sec, step %d, loss %.2f, vloss %.2f" %(elapsed, step, loss0,
                                                     vloss0))

    # todo: factor out buffering into util
    writer.write('%d, %f, %f, %f\n'%(step, elapsed, loss0, vloss0))

    if args.method=='kfac':
      kfac.model.advance_batch()
      kfac.update_stats()

    with u.timeit("train"):
      model.advance_batch()
      grad.update()
      grad_new.update()
      train_op.run()
      u.record_time()

  u.summarize_time()
  sw.close()
  
  if args.mode == 'record':
    u.dump_with_prompt(losses, losses_fn)

  elif args.mode == 'test':
    targets = np.loadtxt(release_test_fn, delimiter=",")
    u.check_equal(losses, targets, rtol=1e-2)
    u.summarize_difference(losses, targets)

