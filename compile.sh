#!/bin/sh
nvcc -x cu -arch=compute_60 -code=sm_60 -dc --compiler-options '-fPIC -m64' cuda_exp.cu -I"/usr/include/python3.4" -I/opt/intel/mkl/include -I"/usr/local/cuda/include" -I"/home/chemadmin/.local/lib/python3.4/site-packages/numpy/core/include" -o build/cuda_exp.o
nvcc -shared -arch=compute_60 -code=sm_60 -Xlinker \"--no-as-needed\" build/cuda_exp.o -L"/usr/lib/python3.4" -L/opt/intel/mkl/lib/intel64 -L"/usr/local/cuda/lib64" -lcublas -lcublas_device -lcudadevrt -lcudart_static -lcuda -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lmkl_vml_avx2 -o build/wordtovec.so
