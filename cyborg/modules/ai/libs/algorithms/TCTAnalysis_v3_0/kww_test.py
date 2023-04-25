#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 23:23
# @Author  : Can Cui
# @File    : test.py
# @Software: PyCharm
# @Comment:
import sys
import os
from settings import *
sys.path.insert(0, lib_path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Slide.dispatch import openSlide
from Algorithms.TCTAnalysis_v3_0.tct_alg import *
import time



# from multiprocessing import Process
# import subprocess
# import torch
# def f(x):
#     print(torch.cuda.device_count())
tic = time.time()
# slide=  openSlide(r'X:\漏阳1\20220329_192115_1.sdpc')
# slide = openSlide(r'D:\datasets\HSIL-AY-20x-48.sdpc')
slide = openSlide(r'D:\data\bailixin\data\2022_07_17_14_22_00_555254\slices\20220717142200515809\1807759.hdx')
# slide = openSlide(r'./test_data/506445.hdx')

ALG = TCT_2_0_sxfszyy()
result = ALG.cal_tct(slide)

print(time.time()-tic)
print(result)



# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
#     command = ['mpiexec', '-np', '4', 'python', '-m', 'test1']
#
#     command_str = ' '.join(command)
#     print(command_str)
#     status = subprocess.Popen(command_str, cwd=os.path.dirname(__file__), shell=True, env=os.environ.copy())
#     import pdb; pdb.set_trace()
