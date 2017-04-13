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

  if not fs:
    fs = [dsize, 28*28, 196, 28*28]
  if not W0f:
    W0f = W_uniform(fs[2],fs[3])
  rho = tf.constant(rho, dtype=dtype)

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = f(-1)
  n = len(fs) - 2

  init_dict = {}
  def init_var(val, name, trainable=True):
    holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
    var = tf.Variable(holder, name=name+"_var", trainable=trainable)
    init_dict[holder] = val
    return var

  Wf = init_var(W0f, "Wf")
  Wf_copy = init_var(W0f, "Wf_copy")
  W = u.unflatten(Wf, fs[1:])
  X = init_var(X0, "X", False)
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
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  return cost, train_op
  
def complex_train_test():
  
  np.random.seed(0)
  
  do_images = True

  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  dsize = 10000
  patches = train_images[:,:dsize];
  fs = [dsize, 28*28, 196, 28*28]
  cost, train_op = cost_and_grad(fs=fs, X0=patches, lambda_=3e-3, rho=0.1, beta=3, lr=0.1)
  
  sess = tf.get_default_session()


  u.reset_time()
  old_cost = sess.run(cost)
  old_i = 0
  frame_count = 0
  for i in range(1000):
    cost0, _ = sess.run([cost, train_op])
    if i%100 == 0:
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

  
if __name__=='__main__':
  complex_train_test()
