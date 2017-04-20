# conventions. "_op things are ops"
# "x0" means numpy
# _live means it's used to update a variable value
# experiment prefixes
#prefix = "small_final2"
prefix = "linesearch3"

import util
import util as u

use_preconditioner = True
drop_l2 = True
drop_sparsity = True
use_gpu = True
do_line_search = False
intersept_op_creation = False

import sys
whitening_mode = int(sys.argv[1])
whiten_every_n_steps = 1
report_frequency = 10
num_steps = 1000
USE_MKL_SVD=True


# adaptive line search
adaptive_step = False
adaptive_step_frequency = 1
adaptive_step_burn_in = 0 # let optimization go for a bit first

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
  # 64-bit doesn't help much, see
  # https://www.wolframcloud.com/objects/5f297f41-30f7-4b1b-972c-cac8d1f8d8e4
  u.default_dtype = dtype
  machine_epsilon = np.finfo(dtype).eps
  
  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  dsize = 10000
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
  W = u.unflatten(Wf, fs[1:])   # todo perf: get rid of this because transposes
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
  pre_dW_stable = [None]*(n+1)  # preconditioned stable dW

  cov_A = [None]*(n+1)
  cov_B2 = [None]*(n+1)
  vars_svd_A = [None]*(n+1)
  vars_svd_B2 = [None]*(n+1)
  for i in range(1,n+1):
    cov_A[i] = init_var(A[i]@t(A[i])/dsize, "cov_A%d"%(i,))
    cov_B2[i] = init_var(B2[i]@t(B2[i])/dsize, "cov_B2%d"%(i,))
    vars_svd_A[i] = SvdWrapper(cov_A[i],"svd_A_%d"%(i,))
    vars_svd_B2[i] = SvdWrapper(cov_B2[i],"svd_B2_%d"%(i,))
    whitened_A = u.pseudo_inverse2(vars_svd_A[i]) @ A[i]
    whitened_B2 = u.pseudo_inverse2(vars_svd_B2[i]) @ B[i]
    whitened_A_stable = u.pseudo_inverse_sqrt2(vars_svd_A[i]) @ A[i]
    whitened_B2_stable = u.pseudo_inverse_sqrt2(vars_svd_B2[i]) @ B[i]
    pre_dW[i] = (whitened_B2 @ t(whitened_A))/dsize
    pre_dW_stable[i] = (whitened_B2_stable @ t(whitened_A_stable))/dsize
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
  pre_grad_stable_live = u.flatten(pre_dW_stable[1:]) # stable pre gradient
  grad = init_var(grad_live, "grad")
  pre_grad = init_var(pre_grad_live, "pre_grad")
  pre_grad_stable = init_var(pre_grad_stable_live, "pre_grad_stable")

  update_params_op = Wf.assign(Wf-lr*pre_grad).op
  update_params_stable_op = Wf.assign(Wf-lr*pre_grad_stable).op
  save_params_op = Wf_copy.assign(Wf).op
  pre_grad_dot_grad = tf.reduce_sum(pre_grad*grad)
  pre_grad_stable_dot_grad = tf.reduce_sum(pre_grad*grad)
  grad_norm = tf.reduce_sum(grad*grad)
  pre_grad_norm = tf.reduce_sum(pre_grad*pre_grad)

  def dump_svd_info(step):
    """Dump singular values and gradient values in those coordinates."""
    for i in range(1, n+1):
      svd = vars_svd_A[i]
      s0, u0, v0 = sess.run([svd.s, svd.u, svd.v])
      util.dump(s0, "A_%d_%d"%(i, step))
      A0 = A[i].eval()
      At0 = v0.T @ A0
      util.dump(A0 @ A0.T, "Acov_%d_%d"%(i, step))
      util.dump(At0 @ At0.T, "Atcov_%d_%d"%(i, step))
      util.dump(s0, "As_%d_%d"%(i, step))

    for i in range(1, n+1):
      svd = vars_svd_B2[i]
      s0, u0, v0 = sess.run([svd.s, svd.u, svd.v])
      util.dump(s0, "B2_%d_%d"%(i, step))
      B0 = B[i].eval()
      Bt0 = v0.T @ B0
      util.dump(B0 @ B0.T, "Bcov_%d_%d"%(i, step))
      util.dump(Bt0 @ Bt0.T, "Btcov_%d_%d"%(i, step))
      util.dump(s0, "Bs_%d_%d"%(i, step))      
    
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
    """Initialize our SVD to identity matrices."""
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
    pl = tf.placeholder(dtype, shape=(), name="linesearch_p")
    assign_op = Wf.assign(initial_value - direction*step*pl)
    vals = []
    for i in range(num_steps):
      sess.run(assign_op, feed_dict={pl: i})
      vals.append(loss.eval())
    sess.run(Wf.assign(saved_val)) # restore original value
    return vals
    
  for step in range(num_steps):  # todo: rename i to step
    update_covariances()
    if step%whiten_every_n_steps==0:
      update_svds()

    sess.run(grad.initializer)
    sess.run(pre_grad.initializer)    # todo perf: only one is needed
    sess.run(pre_grad_stable.initializer) 
    
    lr0, loss0 = sess.run([lr, loss])
    save_params_op.run()

    # when grad norm<1, Fisher is unstable, switch to Sqrt(Fisher)
    stabilized_mode = grad_norm.eval()<1

    if stabilized_mode:
      update_params_stable_op.run()
    else:
      update_params_op.run()

    loss1 = loss.eval()
    advance_batch()

    # line search stuff
    target_slope = (-pre_grad_dot_grad.eval() if stabilized_mode else
                    -pre_grad_stable_dot_grad.eval())
    target_delta = lr0*target_slope
    actual_delta = loss1 - loss0
    actual_slope = actual_delta/lr0
    slope_ratio = actual_slope/target_slope  # between 0 and 1.01 

    if do_line_search:
      vals1 = line_search(Wf_copy, pre_grad, lr/100, 40)
      vals2 = line_search(Wf_copy, grad, lr/100, 40)
      u.dump(vals1, "line1-%d"%(i,))
      u.dump(vals2, "line2-%d"%(i,))
      
    losses.append(loss0)
    step_lengths.append(lr0)
    ratios.append(slope_ratio)

    if step % report_frequency == 0:
      print("Step %d loss %.2f, target decrease %.3f, actual decrease, %.3f ratio %.2f grad norm: %.2f pregrad norm: %.2f"%(step, loss0, target_delta, actual_delta, slope_ratio, grad_norm.eval(), pre_grad_norm.eval()))
    
    # don't shrink learning rate once results are very close to minimum
    if adaptive_step_frequency and adaptive_step and step>adaptive_step_burn_in:
      if slope_ratio < alpha and abs(target_delta)>1e-6 and adaptive_step:
        print("%.2f %.2f %.2f"%(loss0, loss1, slope_ratio))
        print("Slope optimality %.2f, shrinking learning rate to %.2f"%(slope_ratio, lr0*beta,))
        sess.run(vard[lr].setter, feed_dict={vard[lr].p: lr0*beta})
        
      # grow learning rate, .99 was ideal for gradient
      elif step>0 and i%50 == 0 and slope_ratio>0.90 and adaptive_step:
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
