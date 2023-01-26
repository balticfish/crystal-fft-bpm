import sys
import os
XCr_path = os.getcwd()+'/../'
sys.path.append(XCr_path)
import time
import multiprocessing
import numpy as np
from XCrystal import *
import XCrTools as tool

comment = 'angle_C_333_12800_eV_waist_500um_crystal_thikcness100um'

print('Running simulation for: ', comment)

sys.path.append(XCr_path)

def single_realization(delta_theta):
    
    xcr = XCrystal(XCr_path+'/config/C333_Angle12p8keV_waist500um.yaml')
    xcr.configure(delta_theta)
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
        
    delta_theta = np.linspace(4e-6,15e-6, Npoints)    
    
    full_run = np.zeros((n_cycle, n_par, 3))
        
    for i in range(0, n_cycle):
        
        print('Cycle number: ', i)
        res = mp_launcher(n_par, delta_theta[i*n_par:i*n_par + n_par])
        full_run[i, :, :] = res
    
    filename = 'run_' + str(n_cycle) + '_' + str(n_par) + '_' + comment
    np.save(filename, full_run)
    
    t1 = time.time()  
    print('Time (s):', t1 - t0)
