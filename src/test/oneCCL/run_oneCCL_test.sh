#!/bin/bash

mpiexec -n 2 -ppn 2 --pmi=pmix --cpu-bind list:1:8 python test_oneCCL.py 
