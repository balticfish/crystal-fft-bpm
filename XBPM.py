import numpy as np
from scipy.special import jv
import time
import XCrTools as tools


class XBPM:

    def __init__(self, XCr):
        
        self.xtools = tools.XCrTools(XCr)
        if (XCr.field is None):
            self.E_in = self.xtools.Gaussian_2D(XCr)
        else:
            self.E_in = XCr.field
            XCr.qprint('Running with imported field')
        self.u = XCr.u
        
        ix, iy = np.shape(self.E_in)
        
        self.U1_store = np.zeros((ix, iy, XCr.M_store+1), dtype=complex)
        self.U2_store = np.zeros((ix, iy, XCr.M_store+1), dtype=complex)

    
    def ksih0_select(self, XCr, params):
        
        u, d_i = params
        jv0_pp = jv(0, 2.0 * XCr.ele_susceptH * XCr.Z / 2.0 / XCr.cosa * d_i) * np.exp((XCr.ele_susceptH * XCr.Z * d_i / 2.0 / XCr.cosa)**2 / 2.0)
        tr = np.ones_like(XCr.Xx, dtype=np.complex)
        idx_nonzero = self.log_h1h2(XCr, u)
        tr[idx_nonzero] = jv0_pp 
        
        return tr 
    
    def ksih1_p_select(self, XCr, params):
        
        u, d_i = params
        jv_pp = jv(1, 2.0 * XCr.ele_susceptH * XCr.Z / 2.0 / XCr.cosa * d_i) 
        tr = np.zeros_like(XCr.Xx, dtype=np.complex)
        idx_nonzero = self.log_h1h2(XCr, u)
        tr[idx_nonzero] = 1j * jv_pp * np.exp(2j * XCr.k0 * u) 

        return tr 
    
    def ksih1_m_select(self, XCr, params):
        
        u, d_i = params
        jv_pm = jv(1, 2.0 * XCr.ele_susceptH * XCr.Z / 2.0 / XCr.cosa * d_i) 
        tr = np.zeros_like(XCr.Xx, dtype=np.complex)
        idx_nonzero = self.log_h1h2(XCr, u)
        tr[idx_nonzero] = 1j * jv_pm * np.exp(-2j * XCr.k0 * u) 
        
        return tr 
    
    
    def Dz0(self, XCr, params):
        
        u, d_i = params

        return np.exp(1j * XCr.Z / 2.0 / XCr.cosa * self.epsxh0x(XCr, u) * d_i) # 'potential' part of propagator related to average susceptibility
    
    
    def epsxh0x(self, XCr, params): 
        
        u = params
        
        return XCr.epsxh0 * self.log_h1h2(XCr, u) # function that  defines distribution of average susceptibility
    
    
    def log_h1h2(self, XCr, params): 
        
        u = params
        
        return (XCr.Xx >= (-XCr.HH + u)) * (XCr.Xx <=(XCr.HH + u)) # function tht defines lower and upper crystal's surfaces


    def PpPm(self, XCr, c_i):
        
        log_pR = XCr.Kx**2.0 + XCr.Ky**2.0 < 1.0
        
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
        
        dz_store = self.Dz0(XCr, [self.u[k], d_i]) # stored for later use 
        ksh_store = self.ksih0_select(XCr, [self.u[k], d_i]) # stored for later use 
                
        U1prop *= dz_store
        U2prop *= dz_store

        U1R = U1prop * ksh_store + U2prop * self.ksih1_p_select(XCr, [self.u[k], d_i])        
        U2R = U2prop * ksh_store + U1prop * self.ksih1_m_select(XCr, [self.u[k], d_i])
        

        return U1R, U2R


    def propagate_with_split_operator_BPM(self, XCr, U1R, U2R):
        
        U1 = U1R.copy()
        U2 = U2R.copy()


        for k in range(XCr.M):

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
                self.U1_store[:,:,int(k/XCr.zsep)] = U1
                self.U2_store[:,:,int(k/XCr.zsep)] = U2
        
            
        return U1, U2                
                    
                
            

