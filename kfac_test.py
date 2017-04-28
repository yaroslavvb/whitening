dsize = 1000
Lambda = 1e-3


# Test mocked implementation of KFAC
import load_MNIST

import util as u
import util
from util import t  # transpose

from kfac import Model
from kfac import Kfac

import tensorflow as tf
import numpy as np

purely_linear = False  # convert sigmoids into linear nonlinearities
purely_relu = True     # convert sigmoids into ReLUs

def W_uniform(s1, s2): # uniform weight init from Ng UFLDL
  r = np.sqrt(6) / np.sqrt(s1 + s2 + 1)
  result = np.random.random(2*s2*s1)*2*r-r
  return result


def ng_init(rows, cols):
  # creates uniform initializer using Ng's formula
  r = np.sqrt(6) / np.sqrt(rows + cols + 1)
  result = np.random.random(rows*cols)*2*r-r
  return result.reshape((rows, cols))


def model_creator(batch_size, dtype=np.float32):
  """Create MNIST autoencoder model."""
  
  model = Model()

  # TODO: actually use batch_size
  init_dict = {}   # todo: rename to feed_dict?
  global_vars = []
  local_vars = []
  # TODO: rename to make_var
  def init_var(val, name, is_global=False):
    """Helper to create variables with numpy or TF initial values."""
    if isinstance(val, tf.Tensor):
      if is_global:
        var = tf.get_variable(name=name, initializer=val)
      else:
        var = tf.Variable(val, name=name)
    else:
      val = np.array(val)
      assert u.is_numeric(val), "Unknown type"
      holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
      if is_global:
        var = tf.get_variable(name=name, initializer=holder)
      else:
        var = tf.Variable(holder, name=name)
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
  patches = train_images[:,:dsize];
  fs = [dsize, 28*28, 196, 28*28]
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  n = len(fs) - 2

  X = init_var(patches, "X")
  W = [None]*n
  W.insert(0, X)
  A = [None]*(n+2)
  A[1] = W[0]
  for i in range(1, n+1):
    W[i] = init_var(ng_init(f(i), f(i-1)), "W_%d"%(i,), is_global=True)
    A[i+1] = nonlin(W[i] @ A[i])
    
  err = A[n+1] - A[1]

  B = [None]*(n+1)
  B2 = [None]*(n+1)
  B[n] = err*d_nonlin(A[n+1])
  _sampled_labels_live = tf.random_normal((f(n), f(-1)), dtype=dtype, seed=0)
  _sampled_labels = init_var(_sampled_labels_live, "to_be_deleted")
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
  pre_dW = [None]*(n+1)   # preconditioned dW
  for i in range(1,n+1):
    cov_A[i] = init_var(A[i]@t(A[i])/dsize, "cov_A%d"%(i,))
    cov_B2[i] = init_var(B2[i]@t(B2[i])/dsize, "cov_B2%d"%(i,))
    vars_svd_A[i] = u.SvdWrapper(cov_A[i],"svd_A_%d"%(i,))
    vars_svd_B2[i] = u.SvdWrapper(cov_B2[i],"svd_B2_%d"%(i,))
    whitened_A = u.regularized_inverse2(vars_svd_A[i],L=Lambda) @ A[i]
    whitened_B2 = u.regularized_inverse2(vars_svd_B2[i],L=Lambda) @ B[i]
    dW[i] = (B[i] @ t(A[i]))/dsize
    pre_dW[i] = (whitened_B2 @ t(whitened_A))/dsize

  model.extra['A'] = A
  model.extra['B'] = B
  model.extra['B2'] = B2
  model.extra['cov_A'] = cov_A
  model.extra['cov_B2'] = cov_B2
  model.extra['vars_svd_A'] = vars_svd_A
  model.extra['vars_svd_B2'] = vars_svd_B2
  model.extra['W'] = W
  model.extra['dW'] = dW
  model.extra['pre_dW'] = pre_dW
    
  model.loss = u.L2(err) / (2 * dsize)
  sampled_labels_live = A[n+1] + tf.random_normal((f(n), f(-1)),
                                                  dtype=dtype, seed=0)
  sampled_labels = init_var(sampled_labels_live, "sampled_labels")
  err2 = A[n+1] - sampled_labels
  model.loss2 = u.L2(err2) / (2 * dsize)
  model.global_vars = global_vars
  assert global_vars == W[1:]
  model.trainable_vars = model.global_vars  # TODO: make them different?
  model.local_vars = local_vars

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

  return model

if __name__ == '__main__':
  # TODO: figure out the scopes
  with tf.variable_scope("kfac", reuse=None):
    model = model_creator(dsize) # TODO: share dataset between models?
    tf.get_variable_scope().reuse_variables()
    kfac = Kfac(model_creator)

  sess = tf.InteractiveSession()
  model.initialize_global_vars()
  model.initialize_local_vars() 
  
  kfac.reset()    # resets optimization
  for i in range(100):
    kfac.adaptive_step()
    print("Loss %.2f"%(model.loss.eval()))
  
  pass
