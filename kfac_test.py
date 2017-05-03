LR=0.02
LAMBDA=1e-1
use_tikhonov=False
# Use lambda for small batch
#Lambda = 2*1e-1
#Lambda = 1e-3
#Lambda=1e8
#Lambda = 10

num_steps = 10
use_fixed_labels = True
hack_global_init_dict = {}


prefix="kfac_refactor_test7"

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
#dsize = kfac.dsize

import sys
import tensorflow as tf
import numpy as np

# TODO: get rid of this
purely_linear = False  # convert sigmoids into linear nonlinearities
purely_relu = False     # convert sigmoids into ReLUs
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

  # TODO: actually use batch_size
  init_dict = {}   # todo: rename to feed_dict?
  global_vars = []
  local_vars = []
  
  # TODO: rename to make_var
  def init_var(val, name, is_global=False):
    """Helper to create variables with numpy or TF initial values."""
    if isinstance(val, tf.Tensor):
      var = u.get_variable(name=name, initializer=val, reuse=is_global)
    else:
      val = np.array(val)
      assert u.is_numeric(val), "Unknown type"
      holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
      var = u.get_variable(name=name, initializer=holder, reuse=is_global)
      init_dict[holder] = val

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
    
  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  patches = train_images[:,:batch_size];
  fs = [batch_size, 28*28, 196, 28*28]
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  n = len(fs) - 2

  X = init_var(patches, "X", is_global=False)
  W = [None]*n
  W.insert(0, X)
  A = [None]*(n+2)
  A[1] = W[0]
  W0f_old = W_uniform(fs[2],fs[3]).astype(dtype) # to match previous generation
  W0s_old = u.unflatten(W0f_old, fs[1:])   # perftodo: this creates transposes
  for i in range(1, n+1):
    #    W[i] = init_var(ng_init(f(i), f(i-1)), "W_%d"%(i,), is_global=True)
    W[i] = init_var(W0s_old[i-1], "W_%d"%(i,), is_global=True)
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

  global_init_op = tf.group(*[v.initializer for v in global_vars])
  def initialize_global_vars():
    sess = tf.get_default_session()
    sess.run(global_init_op, feed_dict=init_dict)
  model.initialize_global_vars = initialize_global_vars

  local_init_op = tf.group(*[v.initializer for v in local_vars])
  print(local_vars)
  def initialize_local_vars():
    sess = tf.get_default_session()
    sess.run(X.initializer, feed_dict=init_dict)  # A's depend on X
    sess.run(_sampled_labels.initializer, feed_dict=init_dict)
    sess.run(local_init_op, feed_dict=init_dict)
  model.initialize_local_vars = initialize_local_vars

  hack_global_init_dict = init_dict
  
  return model

if __name__ == '__main__':
  np.random.seed(0)
  tf.set_random_seed(0)

  dsize = 1000
  sess = tf.InteractiveSession()
  model = model_creator(dsize) # TODO: share dataset between models?
  model.initialize_global_vars()
  model.initialize_local_vars()
  
  kfac = Kfac(model_creator, dsize)   # creates another copy of model, initializes
  # local variables

  kfac.model.initialize_global_vars()
  kfac.model.initialize_local_vars()
  kfac.reset()    # resets optimization variables (not model variables)
  kfac.lr.set(LR)
  kfac.Lambda.set(LAMBDA)

  with u.capture_vars() as opt_vars:
    opt = tf.train.AdamOptimizer()
    grads_and_vars = opt.compute_gradients(model.loss,
                                           var_list=model.trainable_vars)
    grad = IndexedGrad.from_grads_and_vars(grads_and_vars)
    grad_new = kfac.correct(grad)
    train_op = opt.apply_gradients(grad_new.to_grads_and_vars())
  [v.initializer.run() for v in opt_vars]
  
  losses = []
  u.record_time()

  for i in range(num_steps):
    loss0 = model.loss.eval()
    losses.append(loss0)
    print("Loss ", loss0)

    kfac.model.advance_batch()
    kfac.update_stats()

    model.advance_batch()
    grad.update()
    grad_new.update()
    train_op.run()
    
    u.record_time()

  if len(sys.argv)>1 and sys.argv[1]=='record':
    u.dump(losses, prefix+"_losses.csv")
    sys.exit()

  u.summarize_time()
  targets = np.loadtxt("data/kfac_refactor_test7_losses.csv", delimiter=",")
  print("Difference is ", np.linalg.norm(np.asarray(losses)-targets))
  result = u.check_equal(losses, targets, rtol=1e-4)
  print("Test passed: %s" % (result,))
  
