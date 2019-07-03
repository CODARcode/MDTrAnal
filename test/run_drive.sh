#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:/opt/adios2/lib/python3.5/site-packages
export PYTHONPATH=${PYTHONPATH}:../lib

mpirun --allow-run-as-root -n 10 python3 ../src/md_compress_adaptive_adios2.py