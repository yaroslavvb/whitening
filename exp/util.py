import tensorflow as tf

default_dtype = tf.float64

import numpy as np
import sys
import tensorflow as tf
import traceback

def concat_blocks(blocks):
  col_dims = np.array([[int(b.shape[1]) for b in row] for row in blocks])
  col_sums = col_dims.sum(1)
  assert (col_sums[0] == col_sums).all()
  row_dims = np.array([[int(b.shape[0]) for b in row] for row in blocks])
  row_sums = row_dims.sum(0)
  assert (row_sums[0] == row_sums).all()
  
  block_rows = [tf.concat(row, axis=1) for row in blocks]
  return tf.concat(block_rows, axis=0)

def concat_blocks_test():
  blocks = [[tf.constant([[1]]), tf.constant([[1,2]])],
            [tf.transpose(tf.constant([[1,2]])), tf.constant([[1,2],[3,4]])]]
  result = concat_blocks(blocks)
  sess = tf.Session()
  result0 = sess.run(result)
  check_equal(result0, [[1, 1, 2], [1, 1, 2], [2, 3, 4]])
  

def pseudo_inverse(mat, eps=1e-10):
  """Computes pseudo-inverse of mat, treating eigenvalues below eps as 0."""
  
  s, u, v = tf.svd(mat)
  eps = 1e-10   # zero threshold for eigenvalues
  si = tf.where(tf.less(s, eps), s, 1./s)
  return u @ tf.diag(si) @ tf.transpose(v)

def symsqrt(mat):
  s, u, v = tf.svd(mat)
  eps = 1e-10   # zero threshold for eigenvalues
  si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
  return u @ tf.diag(si) @ tf.transpose(v)

def pseudo_inverse_sqrt(mat):
  s, u, v = tf.svd(mat)
  eps = 1e-10   # zero threshold for eigenvalues
  si = tf.where(tf.less(s, eps), s, 1./tf.sqrt(s))
  return u @ tf.diag(si) @ tf.transpose(v)


def Identity(n, dtype=default_dtype):
  """Identity matrix of size n."""
  return tf.diag(tf.ones((n,), dtype=dtype))

# partitions numpy array into sublists of given sizes
def partition_np(vec, sizes):
  assert np.sum(sizes) == len(vec)
  splits = []
  current_idx = 0
  for i in range(len(sizes)):
    splits.append(vec[current_idx: current_idx+sizes[i]])
    current_idx += sizes[i]
  assert current_idx == len(vec)
  return splits

def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i + n]

def partition(vec, sizes):
  assert len(vec.shape) == 1
  assert np.sum(sizes) == vec.shape[0]
  splits = []
  current_idx = 0
  for i in range(len(sizes)):
    splits.append(vec[current_idx: current_idx+sizes[i]])
    current_idx += sizes[i]
  return splits

def partition_test():
  vec = tf.constant([1,2,3,4,5])
  sess = tf.Session()
  result = sess.run(partition(vec, [3, 2]))
  check_equal(result[0], [1,2,3])
  assert (result[1] == [4,5]).all()


def v2c(vec):
  """Converts vector to column matrix."""
  return tf.expand_dims(vec, 1)

def v2c_np(vec):
  """Converts vector to column matrix."""
  return np.expand_dims(vec, 1)

def v2r(vec):
  """Converts vector into row matrix."""
  return tf.expand_dims(vec, 0)
  
def c2v(col):
  """Converts vector into row matrix."""
  return tf.reshape(col, [-1])


def unvectorize_np(vec, rows):
  """Turns vectorized version of tensor into original matrix with given
  number of rows."""
  assert len(vec)%rows==0
  cols = len(vec)//rows;
  return np.array(np.split(vec, cols)).T

def unvectorize(vec, rows):
  assert len(vec.shape) == 1
  assert vec.shape[0]%rows == 0
  cols = int(vec.shape[0]//rows) 
  cols = [v2r(v) for v in tf.split(vec, cols)]
  return tf.transpose(tf.concat(cols, 0))

def unvectorize_test():
  vec = tf.constant([1,2,3,4,5,6])
  sess = tf.Session()
  result = sess.run(unvectorize(vec, 2))
  assert (result==[[1,3,5],[2,4,6]]).all()

def vectorize_np(mat):
  return mat.reshape((-1, 1), order="F")

def vectorize(mat):
  """Turns matrix into a column."""
  return tf.reshape(tf.transpose(mat), [-1,1])

def vectorize_test():
  mat = tf.constant([[1, 3, 5], [2, 4, 6]])
  sess = tf.Session()
  check_equal(sess.run(c2v(vectorize(mat))), [1,2,3,4,5,6])


def Kmat(rows, cols):
  """Commutation matrix. Kmat(a,b).vec(M) takes vec of a,b matrix M to vec of
  its transpose."""
  input_mat = np.reshape(np.arange(rows*cols),[rows,-1]).astype(np.int32)
  output_mat = input_mat.T
    
  input_vec = vectorize_np(input_mat)
  output_vec = vectorize_np(output_mat)
    
  K = np.zeros((rows*cols, rows*cols), dtype=np.int32)
  for output_idx in range(rows*cols):
    for input_idx in range(rows*cols):
      K[output_idx, input_idx] = (output_vec[output_idx] == input_vec[input_idx])
  return K

def Kmat_test():
  check_equal(Kmat(3,2),
              [[1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1]])

  check_equal(Kmat(2,3),
              [[1, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 1]])

# turns flattened representation into list of matrices with given matrix
# sizes
def unflatten_np(Wf, fs):
  if len(Wf.shape)==2 and Wf.shape[1] == 1:  # treat col mats as vectors
    Wf = Wf.reshape(-1)
    
  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  assert np.sum(sizes)==len(Wf)
  Wsf = partition(Wf, sizes)
  Ws = [unvectorize_np(Wsf[i], dims[i][0]) for i in range(len(sizes))]
  return Ws

# Turns flattened Tensor into list of rank-2 tensors with given sizes
def unflatten(Wf, fs):
  Wf_shape = fix_shape(Wf.shape)
  if len(Wf_shape)==2 and Wf_shape[1] == 1:  # treat col mats as vectors
    Wf = tf.reshape(Wf, [-1])
  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  assert len(Wf.shape) == 1
  assert np.sum(sizes)==Wf.shape[0]
  Wsf = partition(Wf, sizes)
  Ws = [unvectorize(Wsf[i], dims[i][0]) for i in range(len(sizes))]
  return Ws

def unflatten_test():
  vec = tf.constant(list(range(1, 11)))
  sess = tf.Session()
  fs = [2,2,2,1]
  result = sess.run(unflatten(vec, fs))
  check_equal(result[0], [[1,3],[2,4]])
  check_equal(result[1], [[5,7],[6,8]])
  check_equal(result[2], [[9, 10]])


def check_equal(a, b, rtol=1e-12, atol=1e-12):
  try:
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
  except Exception as e:
    print("Error" + "-"*60)
    for line in traceback.format_stack():
      print(line.strip())
        
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("*** print_tb:")
    traceback.print_tb(exc_traceback, limit=10, file=sys.stdout)
    efmt = traceback.format_exc()
    print(efmt)
    import pdb; pdb.set_trace()

# TensorShape([Dimension(2), Dimension(10)]) => (2, 10)
def fix_shape(tf_shape):
  return tuple(int(dim) for dim in tf_shape)

def kronecker_cols(a, b):
  """Treats rank-1 vectors a, b as columns, returns Kronecker product a x b."""
  
  assert len(a.get_shape())==1, "Input a must be rank-1, got shape %s" %(a.get_shape(),)
  assert len(b.get_shape())==1, "Input b must be rank-1, got shape %s"%(a.get_shape(),)
  segments = []
  for i in range(a.get_shape()[0]):
    segments.append(a[i]*b)
  result_vec = tf.concat(segments, axis=0)
  result_col = tf.expand_dims(result_vec, 1)
  return result_col

def kronecker_cols_test():
  a = tf.constant([1,2])
  b = tf.constant([3,4])
  c = tf.transpose(tf.constant([[3,4,6,8]]))
  sess = tf.Session()
  assert sess.run(tf.equal(kronecker_cols(a, b), c)).all()

def kronecker(A, B):
  """Kronecker product of A,B"""

  Arows, Acols = fix_shape(A.shape)
  C = tf.reshape(A, [-1, 1, 1])*tf.expand_dims(B, 0)
  slices = [C[i] for i in range(Arows*Acols)]
  slices_2d = list(chunks(slices, Acols))  # each chunk has Acols elems
  return concat_blocks(slices_2d)

kr = kronecker

def kronecker_test():
  A0 = [[1,2],[3,4]]
  B0 = [[6,7],[8,9]]
  A = tf.constant(A0)
  B = tf.constant(B0)
  C = kronecker(A, B)
  sess = tf.Session()
  C0 = sess.run(C)
  Ct = [[6, 7, 12, 14], [8, 9, 16, 18], [18, 21, 24, 28], [24, 27, 32, 36]]
  Cnp = np.kron(A0, B0)
  check_equal(C0, Ct)
  check_equal(C0, Cnp)


# def merge_mats(mats):
#   """Merges mxn grid of mats into single matrix."""
#   m = len(mats)
#   n = len(mats[0])
#   for i in range(m):
#     for j in range(n):
#       pass

def col(A,i):
  """Extracts i'th column of matrix A"""
  assert len(A.get_shape())==2
  assert i>=0 and i < A.get_shape()[1]
  return tf.expand_dims(A[:,i], 1)
  
def khatri_rao(A, B):
  """Khatri rao product of matrices A,B"""

  cols = []
  assert len(A.get_shape()) == 2, "A must be rank-1, got shape %s" %(a.get_shape(),)
  assert A.get_shape()[1] == B.get_shape()[1]
  for i in range(A.get_shape()[1]):
    cols.append(kronecker_cols(A[:, i], B[:,i]))
  return tf.concat(cols, axis=1)

def khatri_rao_test():
  A = tf.constant([[1, 2], [3, 4]])
  B = tf.constant([[5, 6], [7, 8]])
  C = tf.constant([[5,12], [7,16], [15,24], [21,32]])
  sess = tf.Session()
  assert sess.run(tf.equal(khatri_rao(A, B), C)).all()
  
def relu_mask(a, dtype=default_dtype):
  from tensorflow.python.ops import gen_nn_ops
  ones = tf.ones(a.get_shape(), dtype=dtype)
  return gen_nn_ops._relu_grad(ones, a)

def relu_mask_test():
  a = tf.constant([-1,0,1,2], dtype=default_dtype)
  sess = tf.Session()
  check_equal(sess.run(relu_mask(a)), [0,0,1,1])

def t(x):
  return tf.transpose(x)


if __name__=='__main__':
  relu_mask_test()
  kronecker_test()
  Kmat_test()
  concat_blocks_test()
  kronecker_cols_test()
  khatri_rao_test()
  partition_test()
  unvectorize_test()
  unflatten_test()
  vectorize_test()
