import numpy as np
import time
import multiprocessing
from XCrystal import *
import sys


comment = 'Silicon_004_9000_eV'

print('Running simulation for: ', comment)


XCr_path = '/global/u2/a/aliaksei/CrystalBPM2/lume-crystal-bpm'
sys.path.append(XCr_path)

def single_realization(delta_theta):
    
    xcr = XCrystal(XCr_path+'/Crystal.yaml')
    xcr.configure(delta_theta)
    xcr.run3D()
    
    return np.asarray([delta_theta, xcr.Reflectivity, xcr.Transmission])


def mp_launcher(n_par, delta_theta):
    
    p = multiprocessing.Pool(n_par)
    Observables_multiple = p.map(single_realization, delta_theta)

    return np.asarray(Observables_multiple)



if __name__ == '__main__':
    
    t0 = time.time()
    
    n_cycle = 1
    n_par = 16
    
    Npoints = n_par * n_cycle
        
    delta_theta = np.linspace(-13.862472049580001e-6, 41.54780686945001e-6, Npoints)    
    
    full_run = np.zeros((n_cycle, n_par, 3))
        
    for i in range(0, n_cycle):
        
        print('Cycle number: ', i)
        res = mp_launcher(n_par, delta_theta[i*n_par:i*n_par + n_par])
        full_run[i, :, :] = res
    
    filename = 'run_' + str(n_cycle) + '_' + str(n_par) + '_' + comment
    np.save(filename, full_run)
    
    t1 = time.time()  
    print('Time (s):', t1 - t0)