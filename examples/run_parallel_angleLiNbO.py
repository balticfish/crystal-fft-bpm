import sys
import os
XCr_path = os.getcwd()+'/../'
sys.path.append(XCr_path)
import time
import multiprocessing
import numpy as np
from XCrystal import *
import XCrTools as tool

comment = 'angleLiNbO_14keVZ0p000500alpha_0em3d40grid8001waist0p2xmax0p5umxs0FR'

print('Running simulation for: ', comment)

def single_realization(Rock_angle):
    
    xcr = XCrystal(XCr_path+'/config/CrystalLiNbO.yaml')
    xcr.configure(-0e-3,Rock_angle)
    xcr.run3D()
    
    return np.asarray([Rock_angle, xcr.Reflectivity, xcr.Transmission])

def mp_launcher(n_par,Rock_angle):
    
    p = multiprocessing.Pool(n_par)
    Observables_multiple = p.map(single_realization, Rock_angle)

    return np.asarray(Observables_multiple)

if __name__ == '__main__':
    
    t0 = time.time()
    
    n_cycle = 8
    n_par = 16
    
    Npoints = n_par * n_cycle
        
    Rock_angle = np.linspace(-8e-3,8e-3, Npoints)    
    
    full_run = np.zeros((n_cycle, n_par, 3))
        
    for i in range(0, n_cycle):
        
        print('Cycle number: ', i)
        res = mp_launcher(n_par, Rock_angle[i*n_par:i*n_par + n_par])
        full_run[i, :, :] = res
    
    filename = 'run_' + str(n_cycle) + '_' + str(n_par) + '_' + comment
    np.save(filename, full_run)
    t1 = time.time()  
    print('Time (s):', t1 - t0)