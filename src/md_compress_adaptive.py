import numpy as np
import math

from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import svd_flip, randomized_svd

from scipy.sparse.linalg import svds
from scipy.misc import imfilter
import scipy.ndimage as ndi

import scipy.io

from heapq import heappush, heappop

from random import uniform

import matplotlib
import matplotlib.pyplot as plt


from pprint import pprint

from codar.oas.MDTrSampler import MDTrSampler, svd_wrapper



mat= scipy.io.loadmat('../data/collision_0-5k.mat')
trace = mat['trace']
(x,y,z) = trace.shape
t2 = trace.reshape((x, y*z),order='C')

#Ut, St, VTt = svd_wrapper(t2, 2, method='random')

#c = Ut
#s = St

n_dim=2
step=1000

#fig, ax = plt.subplots()
#ax.scatter(c[:,0],c[:,1], marker=".", c='b', s=0.5)
#ax.set_xlim(min(c[:,0]), max(c[:,0]))
#ax.set_ylim(min(c[:,1]), max(c[:,1]))
#for i in range(0,n,step):
#    ax.annotate(str(i),(c[i,0],c[i,1]))
#plt.show()


#mds =  MDTrSampler(y, n_dim = 2, conv_size = 50, n_samples=100, batch_size=100, manifold_size=64)
#
#nmpsdm, psddu, prob_dist = mds.traj_char(c,s)

#fig_b, ax_b = plt.subplots()
#ax_b.scatter(range(0,nmpsdm.shape[0]),nmpsdm[:,0], marker=".", c="b", s=0.5)
#ax_b.scatter(range(0,psddu.shape[0]),psddu[:],marker="x", c="r", s=0.2)
#plt.show()

#sampling_rate = 0.005
#target, time_stamps = mds.batch_sampling(trace)


#fig_c, ax_c =  plt.subplots()
#ax_c.scatter(range(0,nmpsdm.shape[0]),nmpsdm[:,0], marker=".", c="b", s=0.5)
#ax_c.scatter(time_stamps, np.ones(time_stamps.shape),marker="x",c="r", s=10.0)
#plt.show()


mds2 = MDTrSampler(y, n_dim = 2, conv_size = 50, n_samples=64, batch_size=128, manifold_size=64)
for i in range(512): # not x
    mds2.adaptive_sampling_step(trace[i,:,:])


pprint(mds2.get_sampled_ts())
#pprint(mds2.get_sampled_data())
#pprint(mds2.get_sampled_tsdata())

#fig_d, ax_d = plt.subplots()
#ax_d.scatter(range(0,nmpsdm.shape[0]),nmpsdm[:,0], marker=".", c="b", s=0.5)
#ax_d.scatter(adaptive_samples, np.ones(len(adaptive_samples)),marker="x",c="r", s=10.0)
#plt.show()
