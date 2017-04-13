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

def transposeImageCols(mat):
  """Treats columns of mat as flattened square images, transposes them"""
  result = np.zeros(mat.shape, dtype=mat.dtype)
  w2=mat.shape[0]
  w = int(math.sqrt(w2))
  assert w**2==w2
  for i in range(mat.shape[1]):
    patch=mat[:,i].reshape((w,w))
    result[:,i]=patch.transpose().flatten()
  return result

def W_init(hidden_size, visible_size):
  np.random.seed(0)

  r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
  W1 = np.random.random((hidden_size, visible_size)) * 2 * r - r
  W2 = np.random.random((visible_size, hidden_size)) * 2 * r - r
  return np.concatenate((W1.reshape(hidden_size * visible_size),
                         W2.reshape(hidden_size * visible_size)))


def cost_and_grad(W0f=None, fs=None, lambda_=3e-3, rho=0.1, beta=3,
                  X0=None, lr=0.1):
  """Construct sparse autoencoder loss and gradient.

  Args:
    W0f: initial value of weights (flattened representation)
    fs: list of sizes [dsize, visible, hidden, visible]
    sparsity_param: global feature sparsity target
    beta: weight on sparsity penalty
    X0: value of X (aka W[0])

  Returns:
    cost, train_step
  """

  np.random.seed(0)
  tf.set_random_seed(0)
  dtype = np.float32

  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  dsize = 10000
  patches = train_images[:,:dsize];
  X0 = transposeImageCols(patches)

  if not fs:
    fs = [dsize, 28*28, 196, 28*28]
  if not W0f:
    W0f = W_init(fs[2],fs[3])

  assert X0.shape[1] == fs[0]
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  assert fs[1] == fs[3]
  dsize = f(-1)
  n = len(fs) - 2

  init_dict = {}
  def init_var(val, name, trainable=True):
    holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
    var = tf.Variable(holder, name=name+"_var", trainable=trainable)
    init_dict[holder] = val
    return var

  Wf = init_var(W0f, "Wf")
  W = u.unflatten(Wf, fs[1:])
  X = init_var(X0, "X", False)
  W.insert(0, X)
  #  [W0, W1, W2] = W

  def sigmoid(x):
    return tf.sigmoid(x)
  def d_sigmoid(y):
    return y*(1-y)
  def kl(x, y):
    return x * tf.log(x / y) + (1 - x) * tf.log((1 - x) / (1 - y))
  def d_kl(x, y):
    return (1-x)/(1-y) - x/y
  
  # construct A's
  # A[i] = activations needed to compute gradient of W[i]
  # A[n+1] = network output
  A = [None]*(n+2)
  A[0] = u.Identity(dsize, dtype=dtype)
  A[1] = W[0]
  for i in range(1, n+1): # TODO: prettify
    A[i+1] = sigmoid(tf.matmul(W[i], A[i], name="A"+str(i+1)))
  
  err = (A[3] - A[1])

  # Create B's
  # sparsity penalty
  #  rho_hat = np.sum(a2, axis=1) / m
  # np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).transpose()
  # TODO: replace with keep_dims=True
  rho_hat = v2c(tf.reduce_sum(A[2], axis=1))/dsize

  # B[i] = backprops needed to compute gradient of W[i]
  B = [None]*(n+1)
  
  B[n] = err*d_sigmoid(A[n+1])
  for i in range(n-1, -1, -1):
    backprop = t(W[i+1]) @ B[i+1]
    if i == 1:  # layer one has additional loss term
      backprop += beta*d_kl(rho, rho_hat)
    B[i] = backprop*d_sigmoid(A[i+1])

  # dW[i] = gradient of W[i]
  dW = [None]*(n+1)
  for i in range(n+1):
    # perf todo: replace with transpose inside matmul
    # prettify
    dW[i] = (tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))/dsize +
             lambda_ * W[i])

  # Sparsity
  rho_hat = tf.reduce_sum(A[2], axis=1, keep_dims=True)/dsize
  rho = tf.constant(rho, dtype=dtype)

  # Cost function
  error_term = tf.reduce_sum(tf.square(err)) / (2 * dsize)
  sparsity_term = beta * tf.reduce_sum(kl(rho, rho_hat))
  l2_term = (lambda_ / 2) * (tf.reduce_sum(tf.square(W[1])) + \
                             tf.reduce_sum(tf.square(W[2])))
  cost = error_term  + l2_term + sparsity_term
  
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

  grad = tf.gradients(cost, [Wf])[0]
  grad0f = grad.eval()
  grad0s = u.unflatten_np(grad0f, fs[1:])
  grad0s.insert(0, None)

  mygrad = u.flatten(dW[1:])
  #  u.check_close(grad.eval(), mygrad.eval())  # check_equal works on CPU only

  Wf_copy = init_var(W0f, "Wcopy")
  
  copy_op = Wf_copy.assign(Wf-lr*mygrad)
  with tf.control_dependencies([copy_op]):
    train_op = Wf.assign(Wf_copy)

  sess.run(copy_op)   # because can't initialize and use variable in same .run
  return cost, train_op
  

sess = None

#  min: 9.67, median: 11.55
def simple_train():
  global sess
  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  train_labels = load_MNIST.load_MNIST_labels('data/train-labels-idx1-ubyte')

  # implement MNIST sparse autoencoder
  dsize = 10000
  patches = train_images[:,:dsize];
  X0 = transposeImageCols(patches)
  fs = [dsize, 28*28, 196, 28*28]
  W0f = W_init(fs[2], fs[3])
  init_dict, cost = cost2(W0f, fs, 3e-3, 0.1, 3, X0)

  # for sigmoid use 0.1 lr, for relu use 0.000001
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
  
  #  train_step = tf.train.AdamOptimizer(0.1).minimize(cost)
  do_images = False

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

  u.reset_time()
  old_cost = sess.run(cost)
  old_i = 0
  frame_count = 0
  for i in range(10):
    cost0, _ = sess.run([cost, train_step])
    if i%1 == 0:
      print(cost0)
      # filters are transposed in visualization
    if ((old_cost - cost0)/old_cost > 0.05 or i-old_i>50) and do_images:
      Wf_ = sess.run("Wf_var/read:0")
      W1_ = u.unflatten_np(Wf_, fs[1:])[0]
      display_network.display_network(W1_.T, filename="weights-%03d.png"%(frame_count,))
      frame_count+=1
      old_cost = cost0
      old_i = i
    u.record_time()

  u.summarize_time()

  cost0, _ = profiled_run([cost, train_step])

#  min: 9.67, median: 11.55
def complex_train_test():
  
  do_images = False
  cost, train_op = cost_and_grad()
  sess = tf.get_default_session()
  
  u.reset_time()
  old_cost = sess.run(cost)
  old_i = 0
  frame_count = 0
  for i in range(10):
    cost0, _ = sess.run([cost, train_op])
    if i%1 == 0:
      print(cost0)
      # filters are transposed in visualization
    if ((old_cost - cost0)/old_cost > 0.05 or i-old_i>50) and do_images:
      Wf_ = sess.run("Wf_var/read:0")
      W1_ = u.unflatten_np(Wf_, fs[1:])[0]
      display_network.display_network(W1_.T, filename="pics/weights-%03d.png"%(frame_count,))
      frame_count+=1
      old_cost = cost0
      old_i = i
    u.record_time()

  u.summarize_time()

  #  cost0, _ = profiled_run([cost, train_op])
  print(cost0, (cost0-74.17)*1000)
  assert (cost0-74.17)*1000 < 2  # approximately 17.17

  
if __name__=='__main__':
  complex_train_test()
