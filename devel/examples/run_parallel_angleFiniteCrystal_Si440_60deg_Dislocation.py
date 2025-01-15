import sys
import os
XCr_path = os.getcwd()+'/../'
sys.path.append(XCr_path)
import time
import multiprocessing
import numpy as np
from XCrystal import *
import XCrTools as tool

comment = 'angleFiniteCrystal_Si440_60deg_Dislocation'

print('Running simulation for: ', comment)

def single_realization(delta_theta):
    
    xcr = XCrystal(XCr_path+'/config/Si440_17p45keVDislk60degGronkowskiFiniteCrystal3D.yaml')
    xcr.configure(delta_theta,0)
    xcr.run3D()
    
    return xcr.U1_field, xcr.U2_field
    
def mp_launcher(n_par, delta_theta):
    
    p = multiprocessing.Pool(n_par)
    Observables_multiple = p.map(single_realization,delta_theta )

    return np.asarray(Observables_multiple)

if __name__ == '__main__':
    
    t0 = time.time()
    
    n_cycle = 1
    n_par = 36
    
    Npoints = n_par * n_cycle
        
    delta_theta = np.linspace(-10e-6,10e-6, Npoints)    
    
    #full_run = np.zeros((n_cycle, n_par, 3))
    U1_data = []
    U2_data = []    
    for i in range(0, n_cycle):
        
        print('Cycle number: ', i)
        res = mp_launcher(n_par, delta_theta[i*n_par:i*n_par + n_par])
        U1_data.append(res[:, 0, :, :])
        U2_data.append(res[:, 1, :, :])
    U1_wxyz = np.asarray(U1_data)
    m1, m2, m3, m4,m5 = np.shape(U1_wxyz)
    U2_wxyz = np.asarray(U2_data)

    U1_wxyz = np.reshape(U1_wxyz, (m1*m2, m3, m4, m5))
    U2_wxyz = np.reshape(U2_wxyz, (m1*m2, m3, m4, m5))

    filename1 = 'run_omega_U1' + '_' + str(n_cycle) + '_' + str(n_par) + '_' + comment
    filename2 = 'run_omega_U2' + '_' + str(n_cycle) + '_' + str(n_par) + '_' + comment
    

    np.save(filename1, U1_wxyz)
    np.save(filename2, U2_wxyz)
    
    t1 = time.time()  
    print('Time (s):', t1 - t0)
