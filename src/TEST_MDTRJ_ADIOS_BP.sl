#!/bin/bash -l

#SBATCH -N 2         
#SBATCH -t 00:30:00  
#SBATCH -q regular   
#SBATCH -C knl,quad,cache   
 
srun -n 4 python ./md_compress_adaptive_adios.py &> output.TEST_MDTRJ_ADIOS_BP.$SLURM_JOB_ID


