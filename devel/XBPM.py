'''
This file is part of fft-bpm. It is subject to the license terms in the LICENSE.txt file found in the top-level directory of this distribution and at https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. No part of fft-bpm, including this file, may be copied, modified, propagated, or distributed except according to the terms contained in the LICENSE.txt file.
'''

import numpy as np
from scipy.special import jv
import time
import XCrTools as tools
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

class XBPM:

    def __init__(self, XCr):
        
        self.xtools = tools.XCrTools(XCr)
        if (XCr.field is None):
            self.E_in = self.xtools.Gaussian_2D(XCr)
        else:
            self.E_in = XCr.field
            XCr.qprint('Running with imported field')
        self.u = XCr.u
        
        self.xtools.apply_slit_to_E_in(self, XCr)
        #self.E_in *= ((XCr.Xx - XCr.x00)>= -XCr.slit_x/2)*((XCr.Xx - XCr.x00)<= XCr.slit_x/2)*((XCr.Yy)>= -XCr.slit_y/2)*((XCr.Yy)<= XCr.slit_y/2)
        
        ix, iy = np.shape(self.E_in)
        
        self.U1_store = np.zeros((ix, iy, XCr.M_store+1), dtype=complex)
        self.U2_store = np.zeros((ix, iy, XCr.M_store+1), dtype=complex)

    
    def ksih0_select(self, XCr, params):
        
        d_i,k = params
        jv0_pp1 = jv(0, 2.0 * XCr.ele_susceptH1 * XCr.Z / 2.0 / XCr.cosa * d_i) * np.exp((XCr.ele_susceptH1 * XCr.Z * d_i / 2.0 / XCr.cosa)**2 / 2.0) -1
        jv0_pp2 = jv(0, 2.0 * XCr.ele_susceptH2 * XCr.Z / 2.0 / XCr.cosa * d_i) * np.exp((XCr.ele_susceptH2 * XCr.Z * d_i / 2.0 / XCr.cosa)**2 / 2.0) -1
          
        tr1 =np.zeros_like(XCr.Xx, dtype=complex)
        tr2 = np.zeros_like(XCr.Xx, dtype=complex)
        idx_nonzero1 = self.log_h1h2_film(XCr,k)
        idx_nonzero2 = self.log_h1h2(XCr,k)
        tr1[idx_nonzero1] = jv0_pp1
        tr2[idx_nonzero2] = jv0_pp2
        tr= 1+tr1+tr2

        return tr
 
    
    def ksih1_p_select(self, XCr, params):
        
        d_i,k = params
        jv_pp1 = jv(1, 2.0 * XCr.ele_susceptH1 * XCr.Z / 2.0 / XCr.cosa * d_i) 
        jv_pp2 = jv(1, 2.0 * XCr.ele_susceptH2 * XCr.Z / 2.0 / XCr.cosa * d_i) 
        tr1 = np.zeros_like(XCr.Xx, dtype=complex)
        tr2 = np.zeros_like(XCr.Xx, dtype=complex)
        idx_nonzero1 = self.log_h1h2_film(XCr,k)
        idx_nonzero2 = self.log_h1h2(XCr,k)
        tr1[idx_nonzero1] = 1j * jv_pp1 
        tr2[idx_nonzero2] = 1j * jv_pp2
        tr=tr1+tr2
        return tr 
  
    
    def ksih1_m_select(self, XCr, params):
        
        d_i,k = params
        jv_pp1 = jv(1, 2.0 * XCr.ele_susceptH1 * XCr.Z / 2.0 / XCr.cosa * d_i) 
        jv_pp2 = jv(1, 2.0 * XCr.ele_susceptH2 * XCr.Z / 2.0 / XCr.cosa * d_i) 
        tr1 = np.zeros_like(XCr.Xx, dtype=complex)
        tr2 = np.zeros_like(XCr.Xx, dtype=complex)
        idx_nonzero1 = self.log_h1h2_film(XCr,k)
        idx_nonzero2 = self.log_h1h2(XCr,k)
        tr1[idx_nonzero1] = 1j * jv_pp1 
        tr2[idx_nonzero2] = 1j * jv_pp2
        tr=tr1+tr2 
        return tr 
    
    
    def Dz0(self, XCr, params):
        
        d_i,k = params

        return np.exp(1j * XCr.Z / 2.0 / XCr.cosa * self.epsxh0x(XCr,k) * d_i) # 'potential' part of propagator related to average susceptibility
    
    
    def epsxh0x(self, XCr,params): 
        
        k = params
        
        return (XCr.epsxh01 * self.log_h1h2_film(XCr,k)+XCr.epsxh02 * self.log_h1h2(XCr,k)) # function that  defines distribution of average susceptibility
    
    
    def log_h1h2(self, XCr,params): 
        
        k = params
        
        if XCr.cr_geometry == 'from_file':
            crd = XCr.cr_mask[...,k]
            
        else:    
            crd=(XCr.HH > (np.cos(XCr.asymm_angle)*(XCr.Xx-XCr.xs)-np.sin(XCr.asymm_angle)*XCr.z[k] )) * (-XCr.HH <(np.cos(XCr.asymm_angle)*(XCr.Xx-XCr.xs)-np.sin(XCr.asymm_angle)*XCr.z[k] ))*((XCr.Xx-XCr.xs) >= -XCr.CrSize/2)*((XCr.Xx-XCr.xs) < XCr.CrSize/2)

        return crd
    
    
    def log_h1h2_film(self, XCr,params): 
        
        k = params

        crd=(0 >= (np.cos(XCr.asymm_angle)*(XCr.Xx-XCr.xs+XCr.HH)-np.sin(XCr.asymm_angle)*XCr.z[k] )) * (-XCr.d_film <=(np.cos(XCr.asymm_angle)*(XCr.Xx-XCr.xs+XCr.HH)-np.sin(XCr.asymm_angle)*XCr.z[k] ))*((XCr.Xx-XCr.xs+XCr.HH) >= -XCr.CrSize/2)*((XCr.Xx-XCr.xs+XCr.HH) < XCr.CrSize/2)

        return crd


    def PpPm(self, XCr, c_i):
        
        log_pR = XCr.Kx**2.0 + XCr.Ky**2.0 < 0.1
        
        PMinusR = np.sqrt(1.0 - (XCr.Kx - XCr.k0)**2.0 - XCr.Ky**2.0) * log_pR 
        DkMinusR = np.exp(1j * np.dot(PMinusR, c_i * XCr.Z)) 
        PPlusR = np.sqrt(1.0 - (XCr.Kx + XCr.k0)**2.0 - XCr.Ky**2.0)*log_pR
        DkPlusR = np.exp(1j*np.dot(PPlusR, c_i * XCr.Z))
        
        return PMinusR, DkMinusR, PPlusR, DkPlusR
    
    
    def operator_A(self, U1R, U2R, params):
        
        XCr, c_i = params
        
        PMinusR, DkMinusR, PPlusR, DkPlusR = self.PpPm(XCr, c_i)
        G11 = np.fft.fftshift(self.xtools.fft2(U1R), axes=(0,1))
        G22 = np.fft.fftshift(self.xtools.fft2(U2R), axes=(0,1))
        
        U1prop = self.xtools.ifft2(np.fft.ifftshift(G11 * DkMinusR, axes=(0,1))) 
        U2prop = self.xtools.ifft2(np.fft.ifftshift(G22 * DkPlusR, axes=(0,1))) 

        return U1prop, U2prop
    
    
    def operator_B(self, U1prop, U2prop, params):
        
        XCr, k, d_i = params
        
        dz_store = self.Dz0(XCr,[d_i,k]) # stored for later use 
        ksh_store = self.ksih0_select(XCr,[d_i,k]) # stored for later use 
        #ksih1_m = self.ksih1_p_select(XCr,[d_i,k])* np.exp(2j * XCr.k0 *XCr.Rock_angle*XCr.z1[k])
        #ksih1_p = self.ksih1_p_select(XCr,[d_i,k])* np.exp(-2j * XCr.k0 *XCr.Rock_angle*XCr.z1[k]) 
        #U1R = U1prop * ksh_store*dz_store + U2prop *ksih1_m 
        #U2R = U2prop * ksh_store*dz_store + U1prop *ksih1_p
        ksih1_m = self.ksih1_p_select(XCr,[d_i,k])* np.exp(2j * XCr.k0 *self.u[:,:,k])
        ksih1_p = self.ksih1_p_select(XCr,[d_i,k])* np.exp(-2j * XCr.k0 *self.u[:,:,k])
        U1prop *= dz_store
        U2prop *= dz_store
        U1R = U1prop * ksh_store + U2prop * self.ksih1_p_select(XCr,[d_i,k])* np.exp(2j * XCr.k0 *self.u[:,:,k])       
        U2R = U2prop * ksh_store + U1prop * self.ksih1_m_select(XCr, [d_i,k])* np.exp(-2j * XCr.k0 *self.u[:,:,k])

                
        return U1R, U2R


    def propagate_with_split_operator_BPM(self, XCr, U1R, U2R):
        
        U1 = U1R.copy()
        U2 = U2R.copy()

        plt.show()
        for k in tqdm(range(XCr.M), desc='Beam propagation progress'):

            for i in range(0, len(XCr.c_i)):
                op_cff = len(XCr.c_i) - 1 - i
                
                if XCr.d_i[op_cff] != 0:
                    U1, U2 = self.operator_B(U1, U2, [XCr, k, XCr.d_i[op_cff]])
                else:
                    U1 = U1
                    U2 = U2
                    
                if XCr.c_i[op_cff] != 0:
                    U1, U2 = self.operator_A(U1, U2, [XCr, XCr.c_i[op_cff]])
                else:
                    U1 = U1
                    U2 = U2
                    
            if XCr.store_fields and (k % XCr.zsep == 0):
                self.U1_store[:,:,int(k/XCr.zsep)] = U1 * np.exp(1j * (np.sin(XCr.alpha) - XCr.k0) * XCr.Xx)
                self.U2_store[:,:,int(k/XCr.zsep)] = U2 * np.exp(-1j * (np.sin(XCr.alpha) - XCr.k0) * XCr.Xx)

            plt.show()
            
            
        return U1, U2
                 
                    
                
            

