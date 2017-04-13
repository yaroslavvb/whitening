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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def KL_divergence(x, y):
  return x * tf.log(x / y) + (1 - x) * tf.log((1 - x) / (1 - y))


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

def asciiPlot(mat, threshold=0.1):
  """Returns printable representation of matrix thresholded to binary, use "print" to display """
  binary_array=np.piecewise(mat, [mat < threshold, mat >= threshold], [0, 1]).astype("uint8");
  def stringConvert(arr):
    return ''.join([str(s) for s in arr])
  return '\n'.join([stringConvert(row) for row in list(binary_array)])

def W_init(hidden_size, visible_size):
  np.random.seed(0)

  r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
  W1 = np.random.random((hidden_size, visible_size)) * 2 * r - r
  W2 = np.random.random((visible_size, hidden_size)) * 2 * r - r
  return np.concatenate((W1.reshape(hidden_size * visible_size),
                         W2.reshape(hidden_size * visible_size)))

def cost1(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data0):
  """Construct sparse autoencoder loss, return tensor."""
  
  W10 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size, order='F')
  W20 = theta[hidden_size * visible_size:2 * hidden_size * visible_size].reshape(visible_size, hidden_size, order='F')
  b10 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
  b20 = theta[2 * hidden_size * visible_size + hidden_size:]

  init_dict = {}
  def init_var(val, name, trainable=True):
    holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
    var = tf.Variable(holder, name=name+"_var", trainable=trainable)
    init_dict[holder] = val
    return var

  W1 = init_var(W10, "W1")
  W2 = init_var(W20, "W2")
  b1 = init_var(u.v2c_np(b10), "b1")
  b2 = init_var(u.v2c_np(b20), "b2")
  data = init_var(data0, "data", False)
  
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
    
  # Number of training examples
  m = data0.shape[1]
  a1 = data
  
  # Forward propagation
  z2 = tf.matmul(W1, a1) + b1
  a2 = tf.sigmoid(z2)
  # a2 = tf.nn.relu(z2)
  z3 = tf.matmul(W2, a2) + b2
  a3 = tf.sigmoid(z3)
  #  a3 = tf.nn.relu(z3)
  
  # Sparsity
  rho_hat = tf.reduce_sum(a2, axis=1, keep_dims=True)/m
  rho = tf.constant(sparsity_param, dtype=dtype)

  # Cost function
  cost = tf.reduce_sum(tf.square(a3 - a1)) / (2 * m) + \
          (lambda_ / 2) * (tf.reduce_sum(tf.square(W1)) + \
                           tf.reduce_sum(tf.square(W2))) + \
                           beta * tf.reduce_sum(KL_divergence(rho, rho_hat))
  return init_dict, cost

def cost2(W0f, fs, lambda_, sparsity_param, beta, X0):
  """Construct sparse autoencoder loss and gradient."""

  assert X0.shape[1] == fs[0]
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]

  assert fs[1] == fs[3]
  dsize = f(-1)
  
  init_dict = {}
  def init_var(val, name, trainable=True):
    holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
    var = tf.Variable(holder, name=name+"_var", trainable=trainable)
    init_dict[holder] = val
    return var

  Wf = init_var(W0f, "Wf")
  X = init_var(X0, "X", False)
  W = u.unflatten(Wf, fs[1:])
  # rename data to X
  W.insert(0, X0)
  [W0, W1, W2] = W

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
    
  # Number of training examples
  a1 = X
  
  # Forward propagation
  z2 = tf.matmul(W1, a1)
  a2 = tf.sigmoid(z2)
  z3 = tf.matmul(W2, a2)
  a3 = tf.sigmoid(z3)
  
  # Sparsity
  rho_hat = tf.reduce_sum(a2, axis=1, keep_dims=True)/dsize
  rho = tf.constant(sparsity_param, dtype=dtype)

  # Cost function
  error_term = tf.reduce_sum(tf.square(a3 - a1)) / (2 * dsize)
  sparsity_term = beta * tf.reduce_sum(KL_divergence(rho, rho_hat))
  l2_term = (lambda_ / 2) * (tf.reduce_sum(tf.square(W1)) + \
                             tf.reduce_sum(tf.square(W2)))
  cost = error_term + sparsity_term + l2_term
  return init_dict, cost

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_options.output_partition_graphs = True
run_counter = 0
from tensorflow.python.client import timeline
def profiled_run(tensors):
  """Calls session.run, and saves timeline information to file."""
  global run_counter
  sess = tf.get_default_session()
  run_metadata = tf.RunMetadata()
  result = sess.run(tensors, run_metadata=run_metadata,
                    options=run_options)
  
  tl = timeline.Timeline(run_metadata.step_stats)
  ctf = tl.generate_chrome_trace_format()
  with open('timeline-%d.json'%(run_counter,), 'w') as f:
    f.write(ctf)
  with open('stepstats-%d.pbtxt'%(run_counter,), 'w') as f:
    f.write(str(run_metadata))
  run_counter+=1
  return result


def manual_gradient_test():
  """Construct sparse autoencoder loss and gradient."""

  np.random.seed(0)
  tf.set_random_seed(0)
  dtype = np.float64

  lambda_, sparsity_param, beta = 3e-3, 0.1, 3
  rho = sparsity_param

  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  train_labels = load_MNIST.load_MNIST_labels('data/train-labels-idx1-ubyte')

  dsize = 10000
  patches = train_images[:,:dsize];
  X0 = transposeImageCols(patches)
  fs = [dsize, 28*28, 196, 28*28]
  # W0f = W_init(fs[2], fs[3])
  # W0f = np.random.random((fs[1]*fs[2]+fs[2]*fs[3],))
  # W0f = np.ones((fs[1]*fs[2]+fs[2]*fs[3],), dtype=dtype)
  opttheta = scipy.io.loadmat("opttheta.mat")["opttheta"].flatten().astype(dtype)
  W0f = opttheta[:fs[1]*fs[2]+fs[2]*fs[3]]

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
  [W0, W1, W2] = W

  # layer nonlinearity + its derivative
  def g(t):
    return tf.sigmoid(t)
  def gp(t): # derivative of non-linearity in terms of its output
    return t*(1-t)
    
  # construct A's
  A = [None]*(n+2)
  A[0] = u.Identity(dsize, dtype=dtype)
  A[1] = W0
  for i in range(1, n+1):
    A[i+1] = g(tf.matmul(W[i], A[i], name="A"+str(i+1)))

  # A[2] = g(W1*A1)
  # A[3] = g(W2*A2) = output
  # A[n+1] = network output
  # A[1] = X
  # A[i] activations needed to compute gradient of W[i]
  
  err = (A[1] - A[3])

  # Create B's
  # sparsity penalty
  #  rho_hat = np.sum(a2, axis=1) / m
  # np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).transpose()
  rho_hat = v2c(tf.reduce_sum(A[2], axis=1))/dsize

  # TODO: factor out into dKL
  #  B[1] = B[1] + beta*((1-rho)/(1-rho_hat) - rho/rho_hat)
  B = [None]*(n+1)
  
  B[n] = -err*gp(A[n+1])  # todo: change error to get rid of -
  for i in range(n-1, -1, -1):
    if i == 1:
      B[i] = (t(W[i+1]) @ B[i+1]+beta*((1-rho)/(1-rho_hat) - rho/rho_hat)) * gp(A[i+1])
    else:
      B[i] = (t(W[i+1]) @ B[i+1]) * gp(A[i+1])



  # a's are correct, b/delta are off by one

  # B[1] = t(W[2]) @ B[2] * gp(2)
  # B[0] = t(W[1]) @ B[1] * gp(1)
  # B[i] backprops needed to compute gradient of W[i]

  dW = [None]*(n+1)
  for i in range(n+1):
    # perf todo: replace with transpose inside matmul
    dW[i] = (tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))/dsize +
             lambda_ * W[i])
  # Number of training examples
  a1 = X
  
  # Forward propagation
  z2 = tf.matmul(W1, a1)
  a2 = tf.sigmoid(z2)
  z3 = tf.matmul(W2, a2)
  a3 = tf.sigmoid(z3)
  
  # Sparsity
  rho_hat = tf.reduce_sum(a2, axis=1, keep_dims=True)/dsize
  rho = tf.constant(sparsity_param, dtype=dtype)

  # Cost function
  error_term = tf.reduce_sum(tf.square(a3 - a1)) / (2 * dsize)
  sparsity_term = beta * tf.reduce_sum(KL_divergence(rho, rho_hat))
  l2_term = (lambda_ / 2) * (tf.reduce_sum(tf.square(W1)) + \
                             tf.reduce_sum(tf.square(W2)))
  cost = error_term  + l2_term + sparsity_term
  
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  print(cost.eval())

  # Check sparsity backprop calculation
  m = dsize
  data = A[1].eval()
  h = A[3].eval()
  rho_hat = np.sum(A[2].eval(), axis=1)
  z3 = z3.eval()
  z2 = z2.eval()
  W2 = W[2].eval()
  
  rho = np.tile(sparsity_param, fs[2])
  sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).transpose()
  delta3 = -(data - h) * sigmoid_prime(z3)
  delta2 = (W2.transpose().dot(delta3) + beta * sparsity_delta) * sigmoid_prime(z2)
  #  u.check_equal(delta3, B[2])
  assert u.frobenius_np(delta3-B[2].eval())<1e-5

  grad = tf.gradients(cost, [Wf])[0]
  grad0f = grad.eval()
  grad0s = u.unflatten_np(grad0f, fs[1:])
  grad0s.insert(0, None)

  Wf2 = sess.run(Wf-grad)
  print(grad.eval())

  # STOP HERE
  #  rho_hat = np.sum(a2, axis=1) / m
  #  rho = np.tile(sparsity_param, hidden_size)
  #  sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).T / m


  delta3 = (a3-a1)*a3*(1-a3)/dsize
  delta2 = (t(W2) @ delta3) * a2 * (1 - a2)
  #  delta2 = (W2.T.dot(delta3)) * a2 * (1 - a2)
  W1grad = delta2 @ t(a1) + lambda_ * W1
  W2grad = delta3 @ t(a2) + lambda_ * W2
  print("Error 1")
  #  print(np.sum(np.square((W1grad.eval() - grad0s[1]))))
  print("Error 2")
  print(np.sum(np.square((W2grad.eval() - grad0s[2]))))
  #  assert u.frobenius_np((W1grad.eval() - grad0s[1]))<1e-6
  #  assert u.frobenius_np((W2grad.eval() - grad0s[2]))<1e-5
  u.check_close(a3, A[3])
  u.check_close(a2, A[2])
  u.check_close(a1, A[1])
  #  u.check_close(delta3, B[2])
  #  u.check_close(delta2, B[1])
  #  u.check_close(W1grad, dW[1])
  #  u.check_close(W2grad, dW[2])
  u.check_equal(grad0s[1], dW[1])
  u.check_equal(grad0s[2], dW[2])

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
  [W0, W1, W2] = W

  # TODO: rename to sigmoid, d_sigmoid
  # layer nonlinearity + its derivative
  def g(t):
    return tf.sigmoid(t)
  def gp(t): # derivative of non-linearity in terms of its output
    return t*(1-t)
    
  # construct A's
  # A[i] = activations needed to compute gradient of W[i]
  # A[n+1] = network output
  A = [None]*(n+2)
  A[0] = u.Identity(dsize, dtype=dtype)
  A[1] = W0
  for i in range(1, n+1): # TODO: prettify
    A[i+1] = g(tf.matmul(W[i], A[i], name="A"+str(i+1)))
  
  err = (A[1] - A[3])

  # Create B's
  # sparsity penalty
  #  rho_hat = np.sum(a2, axis=1) / m
  # np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).transpose()
  # TODO: replace with keep_dims=True
  rho_hat = v2c(tf.reduce_sum(A[2], axis=1))/dsize

  # TODO: factor out into dKL
  # B[i] backprops needed to compute gradient of W[i]
  B = [None]*(n+1)
  
  B[n] = -err*gp(A[n+1])  # todo: change error to get rid of -
  for i in range(n-1, -1, -1):
    if i == 1: # todo: split out cost
      B[i] = (t(W[i+1]) @ B[i+1]+beta*((1-rho)/(1-rho_hat) - rho/rho_hat)) * gp(A[i+1])
    else:
      B[i] = (t(W[i+1]) @ B[i+1]) * gp(A[i+1])

  # dW[i] = gradient of W[i]
  dW = [None]*(n+1)
  for i in range(n+1):
    # perf todo: replace with transpose inside matmul
    # prettify
    dW[i] = (tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))/dsize +
             lambda_ * W[i])
  # Number of training examples
  a1 = X
  
  # Forward propagation
  z2 = tf.matmul(W1, a1)
  a2 = tf.sigmoid(z2)
  z3 = tf.matmul(W2, a2)
  a3 = tf.sigmoid(z3)
  
  # Sparsity
  rho_hat = tf.reduce_sum(a2, axis=1, keep_dims=True)/dsize
  rho = tf.constant(rho, dtype=dtype)

  # Cost function
  error_term = tf.reduce_sum(tf.square(a3 - a1)) / (2 * dsize)
  sparsity_term = beta * tf.reduce_sum(KL_divergence(rho, rho_hat))
  l2_term = (lambda_ / 2) * (tf.reduce_sum(tf.square(W1)) + \
                             tf.reduce_sum(tf.square(W2)))
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
  dtype = np.float32
  
  #  manual_gradient_test()
  #  cost_and_grad()
  complex_train_test()
  sys.exit()
  if len(sys.argv)>1 and sys.argv[1] == 'simple':
    simple_train()
  else:
    complex_train()
