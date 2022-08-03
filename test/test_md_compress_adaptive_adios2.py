#!/usr/bin/env python

from threading import local
import numpy as np
import math
import random

from pprint import pprint
from codar.oas.MDTrSampler import MDTrSampler
import codar.utils.pyAdios as ADIOS

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#bpfile = '../src/nwchem_xyz.bp'
engine = 'BP4'
bpfile = '/Codar/nwchem-1/QA/tests/ethanol/nwchem_xyz.bp'

# For SST engine, replace 'BP' with 'SST'.
print('Start analysis: {}'.format(rank))
reader = ADIOS.pyAdios(engine)
reader.open(bpfile, ADIOS.OpenMode.READ)

# to get available attributes
# attrs = reader.available_attributes()
# pprint(attrs)

# to get available variables
avars = reader.available_variables()
if rank == 0:
    print('Available Variables:')
    pprint(avars)

# Q: in the variables, there are 'NATOMS', 'OFFSET' and 'TATOMS'.
#    is one of them related to natoms? It seems 'TATOMS' matches with the
#    'ntoms', i.e. the length of 1D array.
natoms = avars['SLX'].shape[0]
config_ids = []

for line in open('ethanol.oas'):
    if line.startswith('1'):
        config_ids.append(int(line.split()[2]))
nids = len(config_ids)
# print("NIDS:", nids)
# print("NAtoms:", natoms)

# compute the number of atoms and offset for each rank.
# (special care for the last rank)
localatoms = math.ceil(natoms/size)
# print("LocalAtoms:"+str(rank), localatoms)

localids = math.ceil(nids/size)
# print("LocalIDS:"+str(rank), localids)

if rank == (size - 1):
    localatoms = natoms - localatoms * (size - 1)
    localids = nids - localids * (size - 1)
    # print("Rank:", rank, "LocalAtoms:", localatoms)
    # print("Rank:", rank, "LocalIDS:", localids)
    
offset = math.ceil(natoms/size) * rank
offsetid = math.ceil(nids/size) * rank
# print("Offset"+str(rank), offset)
# print("OffsetID"+str(rank), offsetid)


# create sampler object
mds2 = None
if rank == 0:
    mds2 = MDTrSampler(natoms, 2, conv_size=10, n_samples=32, batch_size=50, manifold_size=25)

# maybe not necessary, just to make sure all ranks get the file (stream) handler
# but, this could block all MPI processors forever, if one of processor fails to get
# stream in SST engine.
comm.Barrier()

# loop over steps
step = reader.current_step()
while step >= 0:
    ids = reader.read_variable('SID', start=[offset], count=[localatoms])
    x_val = reader.read_variable('SLX', start=[offset], count=[localatoms])
    y_val = reader.read_variable('SLY', start=[offset], count=[localatoms])
    z_val = reader.read_variable('SLZ', start=[offset], count=[localatoms])
    c_ids = config_ids[offsetid:offsetid+localids]
    if nids != natoms:
        for i in range(localatoms):
            if ids[i] not in c_ids:
                x_val = np.delete(x_val, i)
                y_val = np.delete(y_val, i)
                z_val = np.delete(z_val, i)
    
    # print("{}:{} {}, {}, {}".format(rank, step, len(x_val), len(y_val), len(z_val)))

    if rank == 0:
        master_x = np.empty(natoms, dtype=np.float64)
        master_y = np.empty(natoms, dtype=np.float64)
        master_z = np.empty(natoms, dtype=np.float64)
    else:
        master_x = None
        master_y = None
        master_z = None

    sendcounts = np.array(comm.gather(localatoms, 0))
    # if rank == 0: print("{}:{} {}".format(rank, step, sendcounts))

    # collect all data
    comm.Gatherv(x_val, (master_x, sendcounts), 0)
    comm.Gatherv(y_val, (master_y, sendcounts), 0)
    comm.Gatherv(z_val, (master_z, sendcounts), 0)
    # if rank == 0: print(len(master_x), len(master_y), len(master_z))

    if rank == 0:
        master_xyz = np.array([master_x, master_y, master_y]).T
        # print(master_xyz.shape)
        # print(master_xyz[:,:])
        mds2.adaptive_sampling_step(master_xyz[:,:])

    step = reader.advance()

reader.close()
comm.Barrier()

if rank == 0:
    adaptive_samples = sorted([
        mds2.strm_smplr.reservoir[i][1][0]
        for i in range(mds2.n_samples)
    ])
    print("Adaptive Samples:")
    pprint(adaptive_samples)