import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
import time
import scipy.fft as sp_fft


class XCrTools:
    
    def __init__(self, XCr):
        
        self.nthread_fft = XCr.nthread_fft
        print('Initialized tools...')
        
        
    def Gaussian_2D(self, XCr):
        '''
        Def. of Gaussian beam with  angle of incidence alpha, waist radius om0,position from the source zX and shifted w/r to origin by x00 and with the fast oscilating component exp(1j*k0*x) removed
        '''
        
        return XCr.E0 * 1.0 / (1.0 + 1j * XCr.zX / XCr.zR) * np.exp(-(((((XCr.Xx - XCr.x00)**2.0 + XCr.Yy**2.0) / ((XCr.om0**2.0) * (1.0 + 1j * XCr.zX / XCr.zR)))))) * np.exp(1j * (np.sin(XCr.alpha) - XCr.k0) * XCr.Xx)    
    
    
    def fft2(self, fftw_in):
        return sp_fft.fftn(fftw_in, axes=(0,1), workers=self.nthread_fft, overwrite_x=False)

    def ifft2(self, ifftw_in):
        return sp_fft.ifftn(ifftw_in, axes=(0,1), workers=self.nthread_fft, overwrite_x=False) 
    
    def intensity(self, U):
        return np.sum(np.abs(U)**2.0)