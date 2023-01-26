import sys
import os
XCr_path = os.getcwd()+'/../'
sys.path.append(XCr_path)
import time
import multiprocessing
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
from XCrystal import *
import XCrTools as tools

#comment = 'Silicon_004_9000_eV'
#comment = 'Diamond_400_9831_eV'
#comment = 'Diamond_333_12800_eV'

comment = 'C333_12800_eVm6p0urad10fstmax1300xgrid2400Corr2waist250zsep100Thickness50um'

print('Running simulation for: ', comment)


#XCr_path = '/global/cscratch1/sd/krzywins/CRYSTALBPM4/crystal-fft-bpm'
sys.path.append(XCr_path)


def single_realization(delta_theta, w_ind):
    
    omega = omega0 + w_sim[w_ind]

    xcr = XCrystal(XCr_path+'/config/C333_Omega12p8keV_waist250um.yaml', omega)
    xcr.configure(delta_theta, E_wxy_sim[w_ind,:,:])
    xcr.run3D()
    
    return xcr.Reflectivity, xcr.Transmission


def single_realization_with_data(w_ind):
    
    omega = omega0 + w_sim[w_ind]

    xcr = XCrystal(XCr_path+'/config/C333_Omega12p8keV_waist250um.yaml', omega)
    xcr.configure(delta_theta, E_wxy_sim[w_ind,:,:])
    xcr.run3D()
    return xcr.U1_field, xcr.U2_field
    #return xcr.U1f, xcr.U2f

def mp_launcher(n_par, w_ind):
    
    p = multiprocessing.Pool(n_par)
    Observables_multiple = p.map(single_realization_with_data, w_ind)

    return np.asarray(Observables_multiple)



if __name__ == '__main__':
    
    t0 = time.time()
    
    #n_cycle = 1
    n_par = 16
    
    # Npoints = n_par * n_cycle
    
    delta_theta = -6e-6
    omega0 = 12800.0

    xcr = XCrystal(XCr_path+'/config//C333_Omega12p8keV_waist250um.yaml', omega0)
    xtools = tools.XCrTools(xcr)

    xcr.tgrid = 2600
    xcr.tmax = 1300 * 1.0e-15
    sigma_t =10 * 1.0e-15
    
    print('Bragg angle: ', xcr.alphaB * 180 / np.pi)
    
    E_txy = np.zeros((xcr.tgrid, xcr.xgrid, xcr.ygrid), dtype=complex)
    t = np.linspace(0, xcr.tmax, xcr.tgrid)
    dt = t[1] - t[0]
    phi = 0 * 2.0*np.pi*((t - xcr.tmax / 2.0) / 2.0)**3
    E0 = np.exp(-(t - xcr.tmax / 2.0)**2 / 2.0 / sigma_t**2) * np.exp(1.j * phi)
    
    for i in range(0, xcr.tgrid):
        E_txy[i, :, :] = E0[i] * xtools.Gaussian_2D(xcr, xcr.alphaB)

    pad_shape = [(xcr.tpad, xcr.tpad), (0, 0), (0, 0)]
    E_txy_padded = xtools.my_pad(E_txy, pad_shape)

    E_wxy_padded = np.fft.fftshift(np.fft.fft(E_txy_padded, axis=0), axes=0) 

    coeff = 2.0 * np.pi * xcr.hbar
    w = np.fft.fftshift(coeff * np.fft.fftfreq(xcr.tgrid + 2 * xcr.tpad, dt)) + 0.2
    
    t_crop = 25
    E_wxy_sim = xtools.crop_3d_wavefront(E_wxy_padded, cropping_factors=[t_crop, 1, 1])
    w_sim = w[int(len(w)/2 - len(w)/t_crop/2):int(len(w)/2 + len(w)/t_crop/2)]
    print(w_sim)
    
    Nscan = len(E_wxy_sim)
    print('Number of scanning points: ', Nscan)
    n_cycle = int(Nscan / n_par)
    print('Number of scanning cycles: ', n_cycle)

    inds = np.arange(0, Nscan+1)

    # Rs = np.zeros(Nscan)
    # Ts = np.zeros(Nscan)
    
    U1_data = []
    U2_data = []

    for i in range(0, n_cycle):
        
        print('Cycle number: ', i)
        res = mp_launcher(n_par, inds[i*n_par:i*n_par + n_par])
        
        U1_data.append(res[:, 0, :, :])
        U2_data.append(res[:, 1, :, :])
        
#        print(np.shape(U1_data))
       
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
    
