import scipy.io
from pprint import pprint
from codar.oas.MDTrSampler import MDTrSampler, svd_wrapper
import unittest

class TestCompress(unittest.TestCase):
    
    def test_a_good_case(self):

      mat= scipy.io.loadmat('data/collision_0-256.mat')
      trace = mat['trace']
      (x,y,z) = trace.shape

      mds2 = MDTrSampler(y, n_dim = 2, conv_size = 50, n_samples=64, batch_size=128, manifold_size=64)
      for i in range(x): # not x
        mds2.adaptive_sampling_step(trace[i,:,:])
      pprint(mds2.get_sampled_ts())
