# n=1000 Times: min: 126.04, median: 132.48
# n=2000 Times: min: 573.03, median: 621.49
# n=4096 Times: min: 5586.02, median: 6032.16
# n=4096 eigh: Times: min: 27758.34, median: 28883.69
# n=4096 xeon v3: Times: min: 7241.70, median: 8477.95
# n=4096 xeon v3, gesvd: min=20487.48, median: 22057.64,
import util
from scipy import linalg  # for svd
import numpy as np

do_symmetric = False

n=4096
x = np.random.randn(n*n).reshape((n,n)).astype(dtype=np.float32)
x = x @ x.T
util.record_time()
for i in range(4):
  if do_symmetric:
    result = linalg.eigh(x)
  else:
    result = linalg.svd(x, lapack_driver='gesvd')
  util.record_time()
util.summarize_time()
