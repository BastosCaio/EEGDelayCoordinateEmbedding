# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:54:41 2022

@author: Workstation-Lab
"""
    
import numpy as np
import pyedflib
import time


import sys
sys.path.append('../')
from dependencies.nolitsa.nolitsa import dimension, delay, utils


file_name = "C:\\Users\\Workstation-Lab\\Documents\\Projetos\\CaioBastos\\Programas\\DelayCoordinateEmbedding\\Python\\AAB_36_GCA_RT_OA_ICA.edf"

f = pyedflib.EdfReader(file_name)
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
        
data_c3 = sigbufs[1,:]

start_proc_time = time.time()
DCE_c3 = utils.reconstruct(data_c3, 5, 67)
end_proc_time = time.time()
time_to_complete = end_proc_time - start_proc_time
print(time_to_complete)


start_proc_time = time.time()
DCE_c3 = utils.reconstructGPU(data_c3, 5, 67)
end_proc_time = time.time()
time_to_complete = end_proc_time - start_proc_time
print(time_to_complete)