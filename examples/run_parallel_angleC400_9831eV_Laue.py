import sys
import os
XCr_path = os.getcwd()+'/../'
sys.path.append(XCr_path)
import time
import multiprocessing
import numpy as np
from XCrystal import *
import XCrTools as tool

comment = 'angleC400_9p8keV_Laue_waist2000um_thick_82p90p00deg'

print('Running simulation for: ', comment)

def single_realization(delta_theta):
    
    xcr = XCrystal(XCr_path+'/config/C400_9p8keV_LaueFig3.yaml')
    xcr.configure(delta_theta,0)
    xcr.run3D()
    
    return np.asarray([delta_theta, xcr.Reflectivity, xcr.Transmission])

def mp_launcher(n_par, delta_theta):
    
    p = multiprocessing.Pool(n_par)
    Observables_multiple = p.map(single_realization, delta_theta)

    return np.asarray(Observables_multiple)

if __name__ == '__main__':
    
    t0 = time.time()
    
    n_cycle = 24
    n_par = 16
    
    Npoints = n_par * n_cycle
        
    delta_theta = np.linspace(-8e-6,8e-6, Npoints)    
    
    full_run = np.zeros((n_cycle, n_par, 3))
        
    for i in range(0, n_cycle):
        
        print('Cycle number: ', i)
        res = mp_launcher(n_par, delta_theta[i*n_par:i*n_par + n_par])
        full_run[i, :, :] = res
    
    filename = 'run_' + str(n_cycle) + '_' + str(n_par) + '_' + comment
    np.save(filename, full_run)
    t1 = time.time()  
    print('Time (s):', t1 - t0)