#-*-coding:utf-8-*-
from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import multiprocessing as mp
import threading as td
import time
import numpy as np
import math
import cmath
np.set_printoptions(suppress=True)

def normal():
    res = 0
    shuzu=[]
    for _ in range(2):
        for i in range(10000000):
            res += i+i**2+i**3
	    #print ("IIII")
	    #print (i)
	    shuzu.append(res)
    print('normal:', res)
    print ("normal%e"% res)
    print('shuzulen:', len(shuzu))

def zuikuai():
    shuzu=[]
    for _ in range(2):
        for i in range(10000000):
	    shuzu.append(i)
    a_gpu = gpuarray.to_gpu(np.array(shuzu).astype(np.float32))
    from pycuda.elementwise import ElementwiseKernel
    lin_comb = ElementwiseKernel(
            "float *x,float *z",
            "z[i] = my_f(x[i]+x[i]*x[i]+x[i]*x[i]*x[i])",
            "linear_combination",
            preamble="""
            __device__ float my_f(float x)
            { 
              return x;
            }
            """)
    c_gpu = gpuarray.empty_like(a_gpu)
    lin_comb(a_gpu,c_gpu)


if __name__ == '__main__':
    st = time.time()
    normal()
    st1= time.time()
    print('normal time:', st1 - st)
    st_zuikuai = time.time()
    zuikuai()
    st1_zuikuai= time.time()
    print('zuikuai time:', st1_zuikuai - st_zuikuai)
