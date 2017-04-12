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

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
import util as u


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
  do_images = True

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

  u.reset_time()
  old_cost = sess.run(cost)
  old_i = 0
  frame_count = 0
  for i in range(10000):
    cost0, _ = sess.run([cost, train_step])
    if i%100 == 0:
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

if __name__=='__main__':
  dtype = tf.float32
  
  simple_train()
