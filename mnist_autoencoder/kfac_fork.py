# todo: replace all parts of whitening with svd
# run for longer
# replace svd with scipy svd

use_preconditioner = True
adaptive_step = False
drop_l2 = True
drop_sparsity = True
drop_reconstruction = False
do_single_core = False
use_gpu = False

import sys
#whitening_mode = int(sys.argv[1])
whitening_mode=3
whiten_every_n_steps = 1

natural_samples = 1

"""Do line searches, dump csvs"""

import networkx as nx
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
if use_gpu:
  os.environ['CUDA_VISIBLE_DEVICES']='0'
else:
  os.environ['CUDA_VISIBLE_DEVICES']=''

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

class MyAssign:
  def __init__(self, op, holder=None):
    self.op = op
    self.holder = holder

# variable class that caches assign ops
class MyVar:
  def __init__(self, var):
    self.var = var
    self.assign_ops = {}
    self.placeholder = tf.placeholder(dtype=var.dtype,
                                      shape=var.shape)
    self.placeholder_assign = var.assign(self.placeholder)

  def assign(tensor_or_ndarray):
    if isinstance(tensor_or_ndarray, np.ndarray):
      init_dict = {self.placeholder: tensor_or_ndarray}
      sess.run(self.placeholder_assign, feed_dict=init_dict)
    else:
      if tensor_or_ndarray in self.assign_ops:
        sess.run(self.assign_ops[tensor_or_ndarray])
      else:
        print("Creating new assign for %s=%s"%(var.name,
                                               tensor_or_ndarray.name)) 
        self.assign_ops[tensor_or_ndarray] = var.assign(tensor_or_ndarray)
        sess.run(self.assign_ops[tensor_or_ndarray])
       
      
    
def W_uniform(s1, s2):
  # sample two s1,s2 matrices 
  r = np.sqrt(6) / np.sqrt(s1 + s2 + 1)
  result = np.random.random(2*s2*s1)*2*r-r
  #  u.dump(result, "W2.csv")
  return result


if __name__=='__main__':
  np.random.seed(0)
  tf.set_random_seed(0)
  dtype = np.float32
  u.default_dtype = dtype
  
  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  dsize = 5000
  patches = train_images[:,:dsize];
  fs = [dsize, 28*28, 196, 28*28]

  #  u.dump(patches, "mnist.csv")
  #  sys.exit()

  fs=fs
  X0=patches
  lambda_=3e-3
  rho=0.1
  beta=3
  W0f=None
  
  if not W0f:
    W0f = W_uniform(fs[2],fs[3])
  rho = tf.constant(rho, dtype=dtype)

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = f(-1)
  n = len(fs) - 2

  init_dict = {}
  def init_var(val, name, trainable=False):
    val = np.array(val)
    holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
    var = tf.Variable(holder, name=name+"_var", trainable=trainable)
    init_dict[holder] = val
    return var

  lr = init_var(0.2, "lr")
  Wf = init_var(W0f, "Wf", True)
  Wf_copy = init_var(W0f, "Wf_copy", True)
  #  xyz_Wf = init_var(W0f, "Wf", True)  # flattened parameter vector
  W = u.unflatten(Wf, fs[1:])
  X = init_var(X0, "X")
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

  # A[0] is just for shape checks, assert fail on run
  with tf.control_dependencies([tf.assert_equal(1, 0, message="too huge")]):
    A[0] = u.Identity(dsize, dtype=dtype)
  A[1] = W[0]
  for i in range(1, n+1):
    A[i+1] = sigmoid(W[i] @ A[i])
    

  # reconstruction error and sparsity error
  err = (A[3] - A[1])
  rho_hat = tf.reduce_sum(A[2], axis=1, keep_dims=True)/dsize

  # B[i] = backprops needed to compute gradient of W[i]
  # B2[i] = synthetic backprops for natural gradient
  B = [None]*(n+1)
  B2 = [None]*(n+1)
  B[n] = err*d_sigmoid(A[n+1])
  sampled_labels = tf.Variable(tf.random_normal((f(n), f(-1)),dtype=dtype,seed=0), collections=[])
  B2[n] = sampled_labels*d_sigmoid(A[n+1])
  for i in range(n-1, -1, -1):
    backprop = t(W[i+1]) @ B[i+1]
    backprop2 = t(W[i+1]) @ B2[i+1]
    if i == 1 and not drop_sparsity:
      backprop += beta*d_kl(rho, rho_hat)
      backprop2 += beta*d_kl(rho, rho_hat)
    B[i] = backprop*d_sigmoid(A[i+1])
    B2[i] = backprop2*d_sigmoid(A[i+1])

  # dW[i] = gradient of W[i]
  dW = [None]*(n+1)
  dW2 = [None]*(n+1)
  xyz_dW2 = [None]*(n+1)
  dW3 = [None]*(n+1)
  Acov = [None]*(n+1)
  whitenedA = [None]*(n+1)
  whitenedB = [None]*(n+1)
  whitenedB2 = [None]*(n+1)
  Bcov = [None]*(n+1)    # empirical covariances
  Bcov2 = [None]*(n+1)   # natural gradient sampled covariances
  whitenA = [None]*(n+1)
  whitenB = [None]*(n+1)
  
  xyz_whitenA = [None]*(n+1)
  xyz_whitenB = [None]*(n+1)
  Acov_svd = [None]*(n+1)  # i'th entry contains s,u,v of A[i] covariance
  Bcov2_svd = [None]*(n+1)  # i'th entry contains s,u,v of B2[i] covariance

  
  # covariance matrices
  for i in range(1, n+1):
    pass
    #    cov_A[i] = MyVar(...)

  # TODO: add tiling for natural sampling
  # TODO: start at i=1
  xyz_cov_A = [None]*(n+1)
  xyz_cov_B2 = [None]*(n+1)
  xyz_whiten_A = [None]*(n+1)
  xyz_whiten_B2 = [None]*(n+1)
  vars_svd_A = [None]*(n+1)
  vars_svd_B2 = [None]*(n+1)
  for i in range(1,n+1):
    Acov[i] = A[i]@t(A[i])/dsize  # TODO: replace Acov with A_cov
    Bcov[i] = B[i]@t(B[i])/dsize
    Bcov2[i] = B2[i]@t(B2[i])/(dsize*natural_samples)
    A_len = int(Acov[i].shape[0])
    B2_len = int(Bcov2[i].shape[0])
    xyz_cov_A[i] = tf.Variable(A[i]@t(A[i])/dsize, collections=[])
    xyz_cov_B2[i] = tf.Variable(B2[i]@t(B2[i])/dsize, collections=[])
    xyz_whiten_A[i] = tf.Variable(u.Identity(f(i-1)), "whiten_A[%d]"%(i,))
    xyz_whiten_B2[i] = tf.Variable(u.Identity(f(i)), "whiten_B2[%d]"%(i,))
    init_svd_A = u.Identity(xyz_cov_A[i].shape[0])
    init_svd_B2 = u.Identity(xyz_cov_B2[i].shape[0])  # todo, extend u.Identity
    vars_svd_A[i] = [tf.Variable(tf.ones((A_len,), dtype=dtype)),
                tf.Variable(init_svd_A, dtype=dtype),
                tf.Variable(init_svd_A, dtype=dtype)]
    vars_svd_B2[i] = [tf.Variable(tf.ones((B2_len,), dtype=dtype)),
                 tf.Variable(init_svd_B2, dtype=dtype),
                 tf.Variable(init_svd_B2, dtype=dtype)]
    dW[i] = (B[i]) @ t(A[i])/dsize
    # TODO: rename dW2 to predW
#    xyz_dW2[i] = (xyz_whiten_B2[i] @ B[i]) @ t(xyz_whiten_A[i] @ A[i])/dsize
    xyz_dW2[i] = (u.pseudo_inverse_sqrt2(vars_svd_B2[i]) @ B[i]) @ t(xyz_whiten_A[i] @ A[i])/dsize

  # Cost function
  reconstruction = u.L2(err) / (2 * dsize)
  sparsity = beta * tf.reduce_sum(kl(rho, rho_hat))
  L2 = (lambda_ / 2) * (u.L2(W[1]) + u.L2(W[1]))

  cost = 0
  if not drop_reconstruction:
    cost = cost + reconstruction
  if not drop_l2:
    cost = cost + L2
  if not drop_sparsity:
    cost = cost + sparsity

  grad = u.flatten(dW[1:])    # true gradient
  #  xyz_grad = u.flatten(dW[1:])    # true gradient
  # todo, collapse live/var versions into var.initializer
  xyz_grad_live = u.flatten(dW[1:])
  xyz_pregrad_live = u.flatten(xyz_dW2[1:]) # preconditioned gradient
  
  xyz_grad = tf.Variable(grad)#xyz_grad_live)
  xyz_pregrad = tf.Variable(grad)#xyz_pregrad_live)
  xyz_update_params_op = Wf.assign(Wf-lr*xyz_pregrad).op
  save_params_op = Wf_copy.assign(Wf).op
  xyz_update_grad_op = xyz_grad.assign(xyz_grad_live).op
  xyz_update_pregrad_op = xyz_pregrad.assign(xyz_pregrad_live).op

  if do_single_core:
    sess = tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1,intra_op_parallelism_threads=1))
  else:
    sess = tf.InteractiveSession()

  #  step_len = init_var(tf.constant(0.1), "step_len", False)
  #  step_len_assign = step_len.assign(step_len0)
  step_len0 = tf.placeholder(dtype, shape=())
  
  xyz_pregrad_dot_grad_op = tf.reduce_sum(xyz_pregrad*xyz_grad)
  
  lr_p = tf.placeholder(lr.dtype, lr.shape)
  lr_set = lr.assign(lr_p)

  def advance_batch():
    sess.run(sampled_labels.initializer)  # new labels for next call

  def update_covariances():
    ops_A = [xyz_cov_A[i].initializer for i in range(1, n+1)]
    ops_B2 = [xyz_cov_B2[i].initializer for i in range(1, n+1)]
    sess.run(ops_A+ops_B2)


  # TODO: add names to ops (?) and vars (!)
  sess.run(Wf.initializer, feed_dict=init_dict)
  sess.run(X.initializer, feed_dict=init_dict)
  advance_batch()
  update_covariances()
  init_op = tf.global_variables_initializer()
  sess.run(init_op, feed_dict=init_dict)

  
  print("Running training.")
  do_images = True
  u.reset_time()
  old_cost = sess.run(cost)
  old_i = 0
  frame_count = 0

  step_lengths = []
  costs = []
  ratios = []
  # adaptive line search parameters
  alpha=0.3   # acceptable fraction of predicted decrease
  beta=0.8    # how much to shrink when violation
  growth_rate = 1.05  # how much to grow when too conservative

    
  # todo: use machine precision for epsilon instead of 1e-20
  def xyz_update_cov_A(i):
    sess.run(xyz_cov_A[i].initializer)
  def xyz_update_cov_B2(i):
    sess.run(xyz_cov_B2[i].initializer)

  def update_svd_A(i):
    #  updates SVD parameters of activations covariance matrix a
    (s, u, v) = vars_svd_A[i]
    s1, u1, v1 = tf.svd(xyz_cov_A[i])
    sess.run([s.assign(s1), u.assign(u1), v.assign(v1)])
    #    u0,s0,v0 = cov_A.eval()
    #  ops = [svd_A[i].s.assign, svd_A[i].u_assign, svd_A[i].v_assign]
    #  feed_dict = {svd_A[i].s_holder:s0, svd_A[i].u_holder:u0, svd_A[i].v_holder:v0}
    #  sess.run(ops, feed_dict)

  def update_svd_B2(i):
    #  updates SVD parameters of activations covariance matrix a
    (s, u, v) = vars_svd_B2[i]
    s1, u1, v1 = tf.svd(xyz_cov_B2[i])
    sess.run([s.assign(s1), u.assign(u1), v.assign(v1)])

  def xyz_update_whiten_A(i):
    sess.run(xyz_whiten_A[i].assign(u.pseudo_inverse_sqrt(xyz_cov_A[i])))
    update_svd_A(i)
  def xyz_update_whiten_B2(i):
    sess.run(xyz_whiten_B2[i].assign(u.pseudo_inverse_sqrt(xyz_cov_B2[i])))
    update_svd_B2(i)
  
  def xyz_upgrade_grad():
    sess.run(xyz_grad.initializer)
  def xyz_update_pregrad():
    sess.run(xyz_pregrad.initializer)
    

  if whitening_mode>0:
    xyz_update_whiten_A(1)
    
  def do_line_search(initial_value, direction, step, num_steps):
    saved_val = tf.Variable(Wf)
    sess.run(saved_val.initializer)
    pl = tf.placeholder(dtype, shape=())
    assign_op = Wf.assign(initial_value - direction*step*pl)
    vals = []
    for i in range(num_steps):
      sess.run(assign_op, feed_dict={pl: i})
      vals.append(cost.eval())
    sess.run(Wf.assign(saved_val)) # restore original value
    return vals
    
  for i in range(5):
    sess.run(xyz_update_grad_op)
    sess.run(xyz_update_pregrad_op)

    update_covariances()
    
    lr0, cost0 = sess.run([lr, cost])
    save_params_op.run()
    xyz_update_params_op.run()
    cost1 = cost.eval()

    # advance batch goes here
    advance_batch()
    
    # todo: get rid of expected delta
    xyz_target_delta = -lr0*xyz_pregrad_dot_grad_op.eval()

    actual_delta = cost1 - cost0
    actual_slope = actual_delta/lr0
    xyz_expected_slope = -xyz_pregrad_dot_grad_op.eval()

    # ratio of best possible slope to actual slope
    # don't divide by actual slope because that can be 0
    slope_ratio = abs(actual_slope)/abs(xyz_expected_slope)
    # if slope_ratio>1:
    #   vals1 = do_line_search(Wf_copy, xyz_pregrad, lr/100, 40)
    #   vals2 = do_line_search(Wf_copy, xyz_grad, lr/100, 40)
    #   u.dump(vals1, "line1-%d"%(i,))
    #   u.dump(vals2, "line2-%d"%(i,))
      
    costs.append(cost0)
    step_lengths.append(lr0)
    ratios.append(slope_ratio)

    # TODO: fix B2 labels in variable to avoid recomputing all backprops
    if i%whiten_every_n_steps==0:
      # each is about 200 ms
      if whitening_mode>1:
        xyz_update_whiten_A(2)
      if whitening_mode>2:
        xyz_update_whiten_B2(2)
      if whitening_mode>3:
        xyz_update_whiten_B2(1)

    print("Step %d cost %.2f, target decrease %.3f, actual decrease, %.3f ratio %.2f"%(i, cost0, xyz_target_delta, actual_delta, slope_ratio))
    
    # don't shrink learning rate once results are very close to minimum
    if slope_ratio < alpha and abs(target_delta)>1e-6 and adaptive_step:
      print("%.2f %.2f %.2f"%(cost0, cost1, slope_ratio))
      print("Slope optimality %.2f, shrinking learning rate to %.2f"%(slope_ratio, lr0*beta,))
      sess.run(lr_set, feed_dict={lr_p: lr0*beta})
    else:
      # see if our learning rate got too conservative, and increase it
      # 99 was ideal for gradient
      #      if i>0 and i%50 == 0 and slope_ratio>0.99:
      # todo: replace these values with parameters
      if i>0 and i%50 == 0 and slope_ratio>0.90 and adaptive_step:
        print("%.2f %.2f %.2f"%(cost0, cost1, slope_ratio))
        print("Growing learning rate to %.2f"%(lr0*growth_rate))
        sess.run(lr_set, feed_dict={lr_p: lr0*growth_rate})

    if do_images and i>0 and i%100==0:
      Wf_ = sess.run("Wf_var/read:0")
      W1_ = u.unflatten_np(Wf_, fs[1:])[0]
      display_network.display_network(W1_.T, filename="pics/weights-%03d.png"%(frame_count,))
      frame_count+=1
      old_cost = cost0
      old_i = i


    u.record_time()

  # check against expected loss
  if 'Apple' in sys.version:
    #    u.dump(costs, "mac4.csv")
    targets = np.loadtxt("data/mac4.csv", delimiter=",")
  else:
    #    u.dump(costs, "linux4.csv")
    targets = np.loadtxt("data/linux4.csv", delimiter=",")
    
  u.check_equal(costs[:5], targets[:5])
  u.summarize_time()
