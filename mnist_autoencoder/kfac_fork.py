# conventions. "_op things are ops"
# "x0" means numpy
# _live means it's used to update a variable value
# var, var_set, var_p give variable, assign op, placeholder op

# things to try: get distributions of singular values in each layer
# as well as gradient coordinates as optimization progresses

# todo: run for longer
# todo: replace svd with scipy svd

# experiment prefixes
#prefix="svdtf" # 11:30 Apr 19
prefix="svdmkl"
prefix="svdmkl2"

use_preconditioner = True
adaptive_step = False
drop_l2 = True
drop_sparsity = True
use_gpu = True
do_line_search = False
intersept_op_creation = False

import sys
whitening_mode = int(sys.argv[1])
#whitening_mode=0
whiten_every_n_steps = 10
natural_samples = 1
adaptive_step_frequency = 10
adaptive_step_burn_in = 100 # let optimization go for a bit first
report_frequency = 10

USE_MKL_SVD=True

import networkx as nx
import load_MNIST
import numpy as np
import scipy.io # for loadmat
from scipy import linalg # for svd
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
import util
import util as u
from util import t  # transpose
from util import c2v
from util import v2c
from util import v2c_np
from util import v2r
from util import kr  # kronecker
from util import Kmat # commutation matrix


if intersept_op_creation:
  from tensorflow.python.framework import op_def_library
  old_apply_op = op_def_library.OpDefLibrary.apply_op
  def my_apply_op(obj, op_type_name, name=None, **keywords):
    print(op_type_name+"-"+str(name))
    if op_type_name == 'ExpandDims':
      import pdb; pdb.set_trace()
    return(old_apply_op(obj, op_type_name, name=name, **keywords))
  op_def_library.OpDefLibrary.apply_op=my_apply_op

class VarInfo:
  def __init__(self, setter, p):
    self.setter = setter
    self.p = p

class SvdTuple:
  """Object to store svd tuple.
  Create as SvdTuple((s,u,v)) or SvdTuple(s, u, v).
  """
  def __init__(self, suv, *args):
    if util.list_or_tuple(suv):
      s, u, v = suv
    else:
      s = suv
      u = args[0]
      v = args[1]
      assert len(args) == 2
    self.s = s
    self.u = u
    self.v = v


class SvdWrapper:
  """Encapsulates variables needed to perform SVD of a TensorFlow target.
  Initialize: wrapper = SvdWrapper(tensorflow_var)
  Trigger SVD: wrapper.update_tf() or wrapper.update_scipy()
  Access result as TF vars: wrapper.s, wrapper.u, wrapper.v
  """
  
  def __init__(self, target, name):
    self.name = name
    self.target = target
    self.tf_svd = SvdTuple(tf.svd(target))

    self.init = SvdTuple(
      u.ones(target.shape[0], name=name+"_s_init"),
      u.Identity(target.shape[0], name=name+"_u_init"),
      u.Identity(target.shape[0], name=name+"_v_init")
    )

    assert self.tf_svd.s.shape == self.init.s.shape
    assert self.tf_svd.u.shape == self.init.u.shape
    assert self.tf_svd.v.shape == self.init.v.shape

    self.cached = SvdTuple(
      tf.Variable(self.init.s, name=name+"_s"),
      tf.Variable(self.init.u, name=name+"_u"),
      tf.Variable(self.init.v, name=name+"_v")
    )

    self.s = self.cached.s
    self.u = self.cached.u
    self.v = self.cached.v
    
    self.holder = SvdTuple(
      tf.placeholder(dtype, shape=self.cached.s.shape, name=name+"_s_holder"),
      tf.placeholder(dtype, shape=self.cached.u.shape, name=name+"_u_holder"),
      tf.placeholder(dtype, shape=self.cached.v.shape, name=name+"_v_holder")
    )

    self.update_tf_op = tf.group(
      self.cached.s.assign(self.tf_svd.s),
      self.cached.u.assign(self.tf_svd.u),
      self.cached.v.assign(self.tf_svd.v)
    )

    self.update_external_op = tf.group(
      self.cached.s.assign(self.holder.s),
      self.cached.u.assign(self.holder.u),
      self.cached.v.assign(self.holder.v)
    )

    self.init_ops = (self.s.initializer, self.u.initializer, self.v.initializer)
  

  def update(self):
    if USE_MKL_SVD:
      self.update_scipy()
    else:
      self.update_tf()
      
  def update_tf(self):
    sess = tf.get_default_session()
    sess.run(self.update_tf_op)
    
  def update_scipy(self):
    sess = tf.get_default_session()
    target0 = self.target.eval()
    # A=u.diag(s).v', singular vectors are columns
    u0, s0, vt0 = linalg.svd(target0)
    v0 = vt0.T
    #    v0 = vt0 # bug, makes loss increase, use for sanity checks
    feed_dict = {self.holder.u: u0,
                 self.holder.v: v0,
                 self.holder.s: s0}
    sess.run(self.update_external_op, feed_dict=feed_dict)

    
def W_uniform(s1, s2):
  r = np.sqrt(6) / np.sqrt(s1 + s2 + 1)
  result = np.random.random(2*s2*s1)*2*r-r
  return result


if __name__=='__main__':
  np.random.seed(0)
  tf.set_random_seed(0)
  
  dtype = np.float32
  machine_epsilon = 1e-8  # half distance between 1 and 1+ulp
  u.default_dtype = dtype
  
  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  dsize = 5000
  patches = train_images[:,:dsize];
  fs = [dsize, 28*28, 196, 28*28]

  X0=patches
  lambda_=3e-3
  rho=tf.constant(0.1, dtype=dtype)
  beta=3
  W0f = W_uniform(fs[2],fs[3])

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = f(-1)
  n = len(fs) - 2

  init_dict = {}     # {var_placeholder: init_value}
  vard = {}      # {var: VarInfo}
  def init_var(val, name, trainable=False, noinit=False):
    if isinstance(val, tf.Tensor):
      collections = [] if noinit else None
      var = tf.Variable(val, name=name, collections=collections)
    else:
      val = np.array(val)
      assert u.is_numeric, "Unknown type"
      holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
      var = tf.Variable(holder, name=name, trainable=trainable)
      init_dict[holder] = val
    var_p = tf.placeholder(var.dtype, var.shape)
    var_setter = var.assign(var_p)
    vard[var] = VarInfo(var_setter, var_p)
    return var

  lr = init_var(0.2, "lr")
  Wf = init_var(W0f, "Wf", True)
  Wf_copy = init_var(W0f, "Wf_copy", True)
  W = u.unflatten(Wf, fs[1:])   # todo: get rid of this because transposes
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
  # B2[i] = backprops from sampled labels needed for natural gradient
  B = [None]*(n+1)
  B2 = [None]*(n+1)
  B[n] = err*d_sigmoid(A[n+1])
  sampled_labels_live = tf.random_normal((f(n), f(-1)), dtype=dtype, seed=0)
  sampled_labels = init_var(sampled_labels_live, "sampled_labels", noinit=True)
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
  pre_dW = [None]*(n+1)  # preconditioned dW

  # TODO: add tiling for natural sampling
  cov_A = [None]*(n+1)
  cov_B2 = [None]*(n+1)
  vars_svd_A = [None]*(n+1)
  vars_svd_B2 = [None]*(n+1)
  for i in range(1,n+1):
    cov_A[i] = init_var(A[i]@t(A[i])/dsize, "cov_A%d"%(i,))
    cov_B2[i] = init_var(B2[i]@t(B2[i])/dsize, "cov_B2%d"%(i,))
    vars_svd_A[i] = SvdWrapper(cov_A[i],"svd_A_%d"%(i,))
    vars_svd_B2[i] = SvdWrapper(cov_B2[i],"svd_B2_%d"%(i,))
    whitened_A = u.pseudo_inverse_sqrt2(vars_svd_A[i]) @ A[i]
    # raise epsilon because b's get weird
    whitened_B2 = u.pseudo_inverse_sqrt2(vars_svd_B2[i]) @ B[i]
    if i==1:  # special handling for B2
      whitened_B2 = u.pseudo_inverse_sqrt2(vars_svd_B2[i],eps=1e-3) @ B[i]
    pre_dW[i] = (whitened_B2 @ t(whitened_A))/dsize
    dW[i] = (B[i] @ t(A[i]))/dsize

  # Loss function
  reconstruction = u.L2(err) / (2 * dsize)
  sparsity = beta * tf.reduce_sum(kl(rho, rho_hat))
  L2 = (lambda_ / 2) * (u.L2(W[1]) + u.L2(W[1]))

  loss = reconstruction
  if not drop_l2:
    loss = loss + L2
  if not drop_sparsity:
    loss = loss + sparsity

  grad_live = u.flatten(dW[1:])
  pre_grad_live = u.flatten(pre_dW[1:]) # preconditioned gradient
  grad = init_var(grad_live, "grad")
  pre_grad = init_var(pre_grad_live, "pre_grad")

  update_params_op = Wf.assign(Wf-lr*pre_grad).op
  save_params_op = Wf_copy.assign(Wf).op
  pre_grad_dot_grad = tf.reduce_sum(pre_grad*grad)

  def dump_svd_info(step):
    for i in range(1, n+1):
      s0 = vars_svd_A[i].s.eval()
      u.dump(s0, "A_%d_%d"%(i, step))
      
    
  def advance_batch():
    sess.run(sampled_labels.initializer)  # new labels for next call

  def update_covariances():
    ops_A = [cov_A[i].initializer for i in range(1, n+1)]
    ops_B2 = [cov_B2[i].initializer for i in range(1, n+1)]
    sess.run(ops_A+ops_B2)

  def update_svds():
    if whitening_mode>1:
      vars_svd_A[2].update()
    if whitening_mode>2:
      vars_svd_B2[2].update()
    if whitening_mode>3:
      vars_svd_B2[1].update()

  def init_svds():
    ops = []
    for i in range(1, n+1):
      ops.extend(vars_svd_A[i].init_ops)
      ops.extend(vars_svd_B2[i].init_ops)
    sess = tf.get_default_session()
    sess.run(ops)
      
  init_op = tf.global_variables_initializer()
  tf.get_default_graph().finalize()
  
  sess = tf.InteractiveSession()
  sess.run(Wf.initializer, feed_dict=init_dict)
  sess.run(X.initializer, feed_dict=init_dict)
  advance_batch()
  update_covariances()
  init_svds()
  sess.run(init_op, feed_dict=init_dict)  # initialize everything else
  
  print("Running training.")
  u.reset_time()

  step_lengths = []
  losses = []
  ratios = []
  
  # adaptive line search parameters
  alpha=0.3   # acceptable fraction of predicted decrease
  beta=0.8    # how much to shrink when violation
  growth_rate=1.05  # how much to grow when too conservative
    
  # todo: use machine precision for epsilon instead of 1e-20
  def update_cov_A(i):
    sess.run(cov_A[i].initializer)
  def update_cov_B2(i):
    sess.run(cov_B2[i].initializer)

  # only update whitening matrix of input activations in the beginning
  if whitening_mode>0:
    vars_svd_A[1].update()
    
  def line_search(initial_value, direction, step, num_steps):
    saved_val = tf.Variable(Wf)
    sess.run(saved_val.initializer)
    pl = tf.placeholder(dtype, shape=(), name="linesearch_pl")
    assign_op = Wf.assign(initial_value - direction*step*pl)
    vals = []
    for i in range(num_steps):
      sess.run(assign_op, feed_dict={pl: i})
      vals.append(loss.eval())
    sess.run(Wf.assign(saved_val)) # restore original value
    return vals
    
  for i in range(1000):
    sess.run(grad.initializer)
    sess.run(pre_grad.initializer)
    update_covariances()
    
    if i%whiten_every_n_steps==0:
      update_svds()
    
    lr0, loss0 = sess.run([lr, loss])
    save_params_op.run()
    update_params_op.run()
    loss1 = loss.eval()

    advance_batch()
    
    target_slope = -pre_grad_dot_grad.eval()
    target_delta = lr0*target_slope
    actual_delta = loss1 - loss0
    actual_slope = actual_delta/lr0
    # occasionally see slope ratio 1.01, due to floating point?
    slope_ratio = abs(actual_slope)/abs(target_slope) 

    if do_line_search:
      vals1 = line_search(Wf_copy, pre_grad, lr/100, 40)
      vals2 = line_search(Wf_copy, grad, lr/100, 40)
      u.dump(vals1, "line1-%d"%(i,))
      u.dump(vals2, "line2-%d"%(i,))
      
    losses.append(loss0)
    step_lengths.append(lr0)
    ratios.append(slope_ratio)

    if i % report_frequency == 0:
      print("Step %d loss %.2f, target decrease %.3f, actual decrease, %.3f ratio %.2f"%(i, loss0, target_delta, actual_delta, slope_ratio))
    
    # don't shrink learning rate once results are very close to minimum
    if adaptive_step_frequency and adaptive_step and i>adaptive_step_burnin:
      if slope_ratio < alpha and abs(target_delta)>1e-6 and adaptive_step:
        print("%.2f %.2f %.2f"%(loss0, loss1, slope_ratio))
        print("Slope optimality %.2f, shrinking learning rate to %.2f"%(slope_ratio, lr0*beta,))
        sess.run(vard[lr].setter, feed_dict={vard[lr].p: lr0*beta})
        
      # grow learning rate, .99 was ideal for gradient
      elif i>0 and i%50 == 0 and slope_ratio>0.90 and adaptive_step:
          print("%.2f %.2f %.2f"%(loss0, loss1, slope_ratio))
          print("Growing learning rate to %.2f"%(lr0*growth_rate))
          sess.run(vard[lr].setter, feed_dict={vard[lr].p:
                                               lr0*growth_rate})

    u.record_time()

  # check against expected loss
  if 'Apple' in sys.version:
    #    u.dump(costs, "mac5.csv")
    targets = np.loadtxt("data/mac5.csv", delimiter=",")
  else:
    #    u.dump(costs, "linux5.csv")
    targets = np.loadtxt("data/linux5.csv", delimiter=",")

  u.dump(losses, "%s_losses_%d.csv"%(prefix ,whitening_mode,))
  u.dump(step_lengths, "%s_step_lengths_%d.csv"%(prefix, whitening_mode,))
  u.dump(ratios, "%s_ratios_%d.csv"%(prefix, whitening_mode,))
  u.summarize_time()
