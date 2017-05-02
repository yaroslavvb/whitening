import scipy.linalg as linalg
import numpy as np

target0 = np.fromfile('data/badsvd', np.float32).reshape(784,784)
u0, s0, vt0 = linalg.svd(target0)
