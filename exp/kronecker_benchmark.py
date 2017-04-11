# benchmark Kronecker product
# no shape inference: Constructed 10 x 10 kron 100 times in 2.92 seconds
# Shape inference: Constructed 10 x 10 kron 100 times in 7.95 seconds
# Execution
# single 10x10 kron 100 replicas in .8-1 second

import tensorflow as tf
import util as u
import time
import os
import sys


def benchmark_construct(dims, iters, dtype):
  A = tf.ones((dims, dims), dtype=dtype)
  B = tf.ones((dims, dims), dtype=dtype)
  prods = []
  time0 = time.time()
  for i in range(iters):
    prods.append(u.kr(A,B,True))
  elapsed = time.time() - time0
  print("Constructed %d x %d kron %d times in %.2f seconds"%(A.shape[0], B.shape[0], iters, elapsed))
  
def benchmark_execute(dims, iters, dtype):
  A = tf.random_uniform((dims, dims), dtype=dtype)
  B = tf.random_uniform((dims, dims), dtype=dtype)
  prods = []
  for i in range(iters):
    prods.append(u.kr(A,B,False))
  elapsed_times = []
  sess = tf.Session()
  elapsed_times = []
  for i in range(10):
    time0 = time.time()
    sess.run(tf.group(*prods))
    elapsed_times.append(time.time()-time0)


  print("Executed %d x %d kron %d times in %s seconds"%(A.shape[0], B.shape[0], iters, elapsed_times))
  

if __name__ == '__main__':
  dims = 10
  iters = 100
  dtype = tf.float32
  benchmark_construct(dims, iters, dtype)
  benchmark_execute(dims, iters, dtype)
  
