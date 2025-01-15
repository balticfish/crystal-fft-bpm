'''
This file is part of fft-bpm. It is subject to the license terms in the LICENSE.txt file found in the top-level directory of this distribution and at https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. No part of fft-bpm, including this file, may be copied, modified, propagated, or distributed except according to the terms contained in the LICENSE.txt file.
'''

import numpy as np
from scipy.special import jv
import scipy.constants as sp_const
import yaml
import XBPM as XBPM

class XCrystal:

    def __init__(self, YAML, omega=None, E0=None):
        
        self.config = yaml.load(open(YAML), Loader=yaml.FullLoader)
        
        for key, value in self.config.items():
            setattr(self, key, value)
        
        if (omega is None):
            self.omega = self.omega0
        else:
            self.omega = omega 
            
        self.lam0 = 12398.419300923944 / self.omega0  * 1.0e-10
        self.convr0 = 2.0 * np.pi / self.lam0 # 'convr0' to convert meters to dimensionless units

        self.lam = 12398.419300923944 / self.omega  * 1.0e-10
        self.K0 = 2.0 * np.pi / self.lam #modululs of k vector
        self.K00 = 2.0 * np.pi / self.lam0 #modululs of k vector
        self.convr = self.K0 # 'convr' to convert meters to dimensionless units
        self.convr0 = self.K00 # 'convr' to convert meters to dimensionless units 
        self.c = sp_const.c
        self.hbar = sp_const.hbar / sp_const.e 

        self.a0 = self.config['a0'] * 1.0e-10
        
        self.d = np.sqrt(self.a0**2 / (self.Miller_h**2 + self.Miller_k**2 + self.Miller_l**2)) * self.convr #interplanar spacing
        self.dm = self.d / self.convr #interplanar spacing [m]
        self.asymm_angle = np.deg2rad(self.config['asymm_angle'])
        self.alphaB = np.arcsin(self.lam0 / self.dm / 2.0)

        self.k0 = np.pi / self.d # half of reciprocal vector,  reciprocal vector =  2*k0
        
        self.n_d1 = 1.0 - (self.config['delta1'] -1j * self.config['beta1'])
        self.n_d2 = 1.0 - (self.config['delta2'] -1j * self.config['beta2'])
        
        self.ele_susceptH1 = (self.xrh1 - 1j*self.xih1)
        self.ele_susceptH2 = (self.xrh2 - 1j*self.xih2)
        self.eps1 = self.n_d1**2.0 # average electric susceptibility
        self.eps2 = self.n_d2**2.0 # average electric susceptibility 
        self.epsxh01 = self.eps1 - 1.0 # difference of average electric susceptibility w.r. to vacuum
        self.epsxh02 = self.eps2 - 1.0 # difference of average electric susceptibility w.r. to vacuum
        self.cosa = np.sqrt(1.0 - self.k0**2.0)
        
        self.Z = np.abs(self.Zstep_factor * np.pi/np.real(self.epsxh02))  #  Z - step in z
        
        self.slit_x = self.config['slit_x'] * 1.0e-6 * self.convr/np.cos(self.alphaB)
        self.slit_y = self.config['slit_y'] * 1.0e-6 * self.convr
        
        self.HH = self.config['thickness'] * 1.0e-6 / 2.0 * self.convr
        self.xs = self.config['xs']*1e-6*self.convr
        self.CrSize=self.config['CrSize']*1e-6*self.convr
        
        if self.config['geometry'] == 'from_file':
            self.cr_geometry = 'from_file'
            try:
                self.cr_mask = np.load(self.config['geometry_file'])
                print('Geometry file was provided')
            except:
                print('No geometry file was provided')
        else:
            self.cr_geometry = 'none'

        if (self.quiet_mode == False):
            print('Quiet mode disabled. I will talk a lot...')
        
        
        self.xxmax = self.config['xxmax'] * 1.0e-6 * self.convr # grid size in x
        self.yymax = self.config['yymax'] * 1.0e-6 * self.convr # grid size in y 


        if self.xgrid == 1:
            self.xx=np.array([0])
        else:
             self.xx =  np.linspace(-self.xxmax, self.xxmax, self.xgrid)
            
        if self.ygrid == 1:
            self.yy=np.array([0])
        else:
            self.yy =  np.linspace(-self.yymax, self.yymax, self.ygrid)
            
        
        self.Yy, self.Xx = np.meshgrid(self.yy, self.xx) # x,y mesh/grid
        
        # reciprocal space (angular spectrum)
        self.dkx = 2.0*np.pi/self.xxmax/2.0  # grid resolution,k vector in x
        if self.xgrid == 1:
            self.kkx =np.array([0])
        else:
             self.kkx = self.dkx*(np.arange(1,len(self.xx)+1) - 0.5*len(self.xx))#  k vector in x    
        self.dky = 2.0*np.pi/self.yymax/2.0 # grid resolution,k vector in y 
        if self.ygrid == 1:
            self.kky=np.array([0])
        else:
            self.kky = self.dky*(np.arange(1,len(self.yy)+1) - 0.5*len(self.yy)) #k vector in y
        self.Ky, self.Kx = np.meshgrid(self.kky, self.kkx) # angular spectrum mesh/grid
        
        self.qprint('Congigured grid parameters')
        self.beam = self.config['beam']
        
        if self.beam=='Gaussian':
            self.waist = self.config['waist']
            self.om0 = self.waist * 1e-6 * self.convr   # electric field radius at the waist
            self.omZ = self.waist * 1e-6 * self.convr   # electric field radius at the sample
            self.zR  =  (self.om0**2.0)/2.0     # Rayleigh parameter in the internal units (lamda =2*np.pi)
            self.zX  =  self.zR*np.sqrt(self.omZ**2.0 / self.om0**2.0 - 1.0) #distance of the source w/r to the sample
            if self.config['x00'] == 'auto':
                self.x00 = -3.0 * self.om0 - self.HH # shift  in x w.r. to the origin
            else:
                self.x00 = self.config['x00']*self.convr*1e-6
           
            if (E0 is None):
                self.E0 = self.omZ / self.om0 # amplitude of electric field 
            else:
                self.E0 = E0
            self.qprint('Congigured a Gaussian beam')
            
        if self.beam=='GenesisV2':
            self.fname_GenV2 = self.config['fname_GenV2']
            self.ncar_GenV2 = self.config['ncar_GenV2']
            mag_factor = self.config['mag_factor_GenV2']
            crop_t = self.config['crop_t']
            crop_x = self.config['crop_x']
            self.dgrid_GenV2 = self.config['dgrid_GenV2'] / mag_factor
            self.zsep_GenV2 = self.config['zsep_GenV2']
            self.E0 = 1.0 # amplitude of electric field 
            self.qprint('Congigured a Genesis2 beam')
        
        if self.method == 'Euler':
            
            self.c_i = [1.0]
            self.d_i = [1.0]
            
        if self.method == 'Verlet':
            
            self.d_i = [0.5, 0.5]
            self.c_i = [0.0, 1.0]
            
        if self.method == 'Forest-Ruth':
            
            self.tt = 1.3512
            self.c_i = [0.0, self.tt, 1.0 - 2.0 * self.tt, self.tt]
            self.d_i = [self.tt / 2.0, (1.0 - self.tt) / 2.0, (1.0 - self.tt) / 2.0, self.tt / 2.0]
            
            
    def qprint(self, str): 
        
        if self.quiet_mode:
            pass
        else:
            print(str)
            
            
    def configure(self, Delta_alpha, Rock_angle, field=None):
        
        self.Rock_angle = Rock_angle
        self.Delta_alpha = Delta_alpha
        self.alpha = self.alphaB + self.Delta_alpha
        self.alphaDeg = np.rad2deg(self.alpha)
        
        if (field is None):
            self.field = None
            
        else:
            self.field = field * np.exp(1j * (np.sin(self.alpha) - self.k0) * self.Xx)
            
           
        if self.config['M'] == 'auto':
            self.M = np.int(np.round(0.9*self.xxmax / self.Z / np.tan(self.alpha))) # number of steps in z  
        else:
            self.M= self.config['M']  
 
        self.M_store = int(self.M / self.zsep)
        
        self.deformation_model = self.config['deformation']
        
        self.z0 = np.abs(self.x00-self.HH*0) / np.tan(self.alpha) # approximate positon of reflection center in z-coordinate 
        self.z1=np.linspace(0,self.Z*(self.M-1),self.M)-self.z0
        self.z=np.linspace(0,self.Z*(self.M-1),self.M)-self.Z*self.M/2
        
        #initialize u array with zeros
        self.u = np.zeros((self.xgrid,self.ygrid,self.M))

            
        if self.deformation_model == 'Sinusoidal':
            
            zee = np.arange(1, 1e4+1) * 1e-7 - 0.5e-3 # z-mesh [m]
            lam_z = self.config['def_period'] * 1.0e-6 
            kz = 2.0 * np.pi / lam_z # undulation period [m]
            amp = self.config['def_amplitude'] # undulation amplidude
            uee = amp * np.cos(zee * kz) #undulation function [m]
            ze = zee * self.convr  # undulations in dimensionless units 
            ue = uee * self.convr  # undulations in dimensionless units
            z0 = np.abs(self.x00 - self.HH) / np.tan(self.alpha) # approximate positon of reflection center in z-coordinate 
            z1 = self.Z * np.arange(1, self.M + 1) - z0 # z position vector
            u_z = np.interp(z1, ze, ue) # displacement of crystal planes at z1  
            self.u = self.u_x[np.newaxis, np.newaxis, :]

        
        if self.deformation_model == 'ConstStrainGrad':
            
            self.B1 = self.config['def_B'] / self.convr
            self.u_x = self.B1 * (self.xx + self.HH)**2
            self.u = self.u_x[:, np.newaxis, np.newaxis]
                        
                        
        if self.deformation_model == 'Bent Spectrometer':
            
            self.R = self.config['R'] * self.convr
            self.v = self.config['vC100']
            self.u = (self.Rock_angle * self.z1[np.newaxis, np.newaxis, :] +  1.0 / 2.0 / self.R * (self.v *self.xx[:, np.newaxis, np.newaxis]**2 + self.z1[np.newaxis, np.newaxis, :]**2))*1

                        
        if self.deformation_model == 'Strained film':
            
            self.strain = self.config['strain']
            self.d_film = self.config['d_film']*1e-6*self.convr
            self.u_x = self.strain*(self.xx-self.xs+self.d_film)*((self.xx-self.xs)>=-self.HH-self.d_film)* ((self.xx-self.xs) <= -self.HH)
            self.u = self.Rock_angle * self.z1[np.newaxis, np.newaxis, :] + self.u_x[:, np.newaxis, np.newaxis]

                     
        if self.deformation_model == 'B dopped Diamond Laue':
            
            self.dudz = self.config['dudz']
            self.depth = self.config['depth'] * self.convr * 1e-6
            self.B_size = self.config['B_size'] * self.convr * 1e-6
            
            #CrSize=self.config['CrSize']*1e-6*self.convr
            # u_x = 1/2/R*((v*self.xx)**2+self.z1**2)
        
            # Precompute terms
            X = self.xx - self.x00                  # Shape: (xgrid,)
            X_term = (X ** 4) / (B_size ** 4)       # Shape: (xgrid,)

            Y = self.yy                             # Shape: (ygrid,)
            Y_term = (Y ** 2) / (B_size ** 2)       # Shape: (ygrid,)

            Z = self.z                              # Shape: (M,)
            Z_shifted = Z + self.HH                 # Shape: (M,)
            Z_term = (Z_shifted ** 2) / (self.depth ** 2)  # Shape: (M,)

            # Reshape terms to enable broadcasting
            X_term = X_term[:, np.newaxis, np.newaxis]  # Shape: (xgrid, 1, 1)
            Y_term = Y_term[np.newaxis, :, np.newaxis]  # Shape: (1, ygrid, 1)
            Z_term = Z_term[np.newaxis, np.newaxis, :]  # Shape: (1, 1, M)

            # Sum the terms and compute the exponential
            Exp_term = np.exp(-(X_term + Y_term + Z_term))              # Shape: (xgrid, ygrid, M)

            # Compute the amplitude term
            Amplitude = self.dudz * Z[np.newaxis, np.newaxis, :]  # Shape: (1, 1, M)

            # Compute self.u
            self.u = Amplitude * Exp_term  # Shape: (xgrid, ygrid, M)

                        
                        
                        
        if self.deformation_model == 'Dislocation_60degChukhovskiiConfiguration':
            
            #Silicon b=[110],h=[110], tau=[101], n=[-1,1,1],ix=[1,1,0], jy=[-1,1,-2],kz=[-1,1,1]
            z0 = np.abs(self.x00-self.HH*0) / np.tan(self.alpha) # approximate positon of reflection center in z-coordinate 
            z1 = self.Z * np.arange(1, self.M+1) - z0 # z position vector
            self.x0dm=self.config['x0d']
            phid=self.config['phid']
            slope=self.config['slope']
            burvect_x=self.config['burgers_vector_x']*self.convr*1e-10
            burvect_s=self.config['burgers_vector_screw']*self.convr*1e-10
            
            # Chukhovskii configuration 60 deg
                        
            self.Gronkowski_Chukowski_geometry(ay1=0.8660254, ay2=-0.5, ay3=0, az1=0, az2=0, az3=1)
            
#             # 1. Compute deltas
#             delta_x = self.xx - x0d  # Shape: (xgrid,)
#             delta_y = self.yy - y0d  # Shape: (ygrid,)
#             delta_z = z - z0d        # Shape: (M,)

#             # 2. Reshape for broadcasting
#             delta_x = delta_x[:, np.newaxis, np.newaxis]     # Shape: (xgrid, 1, 1)
#             delta_y = delta_y[np.newaxis, :, np.newaxis]     # Shape: (1, ygrid, 1)
#             delta_z = delta_z[np.newaxis, np.newaxis, :]     # Shape: (1, 1, M)

#             # 3. Compute components A, B, C
#             A = (delta_x * ay1) + (delta_y * ay2) + (delta_z * ay3)  # Shape: (xgrid, ygrid, M)
#             B = (delta_x * az1) + (delta_y * az2) + (delta_z * az3)  # Shape: (xgrid, ygrid, M)
#             C = A + 1j * B  # Complex array

#             # 5. Compute Term1
#             C1 = a0d / (np.sqrt(2) * 2 * np.pi)
#             Term1 = C1 * np.angle(C)  # Shape: (xgrid, ygrid, M)

#             # 6. Compute Term2
#             Numerator = B * A
#             Denominator = A ** 2 + B ** 2
#             Fraction = Numerator / Denominator

#             Term2 = C1 / 2.0 * Fraction / (2 * (1 - v))  # Shape: (xgrid, ygrid, M)

#             # 7. Compute self.u
#             self.u = Term1 + Term2  # Shape: (xgrid, ygrid, M)
            
            # for i in range(self.xgrid):
            #     for j in range(self.ygrid):
            #         for k in range(self.M):
            #             self.u[i,j,k]= 1*a0d/np.sqrt(2)/2/np.pi*(np.angle(((self.xx[i]-x0d)*az1+(self.yy[j]-y0d)*az2+(z[k]-z0d)*az3)*1j+((self.xx[i]-x0d)*ay1+(self.yy[j]-y0d)*ay2+(z[k]-z0d)*ay3)))+1*a0d/np.sqrt(2)/2/np.pi/2*(((self.xx[i]-x0d)*az1+(self.yy[j]-y0d)*az2+(z[k]-z0d)*az3)*((self.xx[i]-x0d)*ay1+(self.yy[j]-y0d)*ay2+(z[k]-z0d)*ay3)/(((self.xx[i]-x0d)*ay1+(self.yy[j]-y0d)*ay2+(z[k]-z0d)*ay3)**2+((self.xx[i]-x0d)*az1+(self.yy[j]-y0d)*az2+(z[k]-z0d)*az3)**2)/(2*(1-v)))          
                        
        if self.deformation_model == 'Dislocation_60degGronkowskiConfiguration':
            
            #i=np.array([1,-1,0])/np.sqrt(2)
            #j=np.array([-1,-1,1])/np.sqrt(3)
            #k=np.array([1,1,2])/np.sqrt(6):h, n
            #i0=([1,1,0])/np.sqrt(2): tau
            #j0=([1,-1,-2])/np.sqrt(6)
            #k0=([1,-1,1])/np.sqrt(3)
            # b=1/2[0,-1,1] from Gronkowski reference
            z0 = np.abs(self.x00-self.HH*0) / np.tan(self.alpha) # approximate positon of reflection center in z-coordinate 
            z1 = self.Z * np.arange(1, self.M+1) - z0 # z position vector
   #         z=np.linspace(0,self.Z*self.M,self.M)-self.Z*self.M/2
            phid=self.config['phid']
            slope=self.config['slope']

            burvect_x=self.config['burgers_vector_x']*self.convr*1e-10
            burvect_s=self.config['burgers_vector_screw']*self.convr*1e-10
            
            #Gronkowski configuration
            

        
            
            #b=[0,1,1]
            
            #ay1=-0.5
            #ay2=-0.40824829
            #ay3=0.57735027
           
            #az1= 0.70710678
            #az2=-0.28867513
            #az3= 0.40824829
            
         
            
            self.Gronkowski_Chukowski_geometry(ay1=-0.5, ay2=0.40824829, ay3=-0.57735027, az1=-0.70710678, az2=-0.28867513, az3=0.40824829)
            

#             # 1. Compute deltas
#             delta_x = self.xx - x0d    # Shape: (xgrid,)
#             delta_y = self.yy - y0d    # Shape: (ygrid,)
#             delta_z = z - z0d          # Shape: (M,)

#             # 2. Reshape for broadcasting
#             delta_x = delta_x[:, np.newaxis, np.newaxis]     # Shape: (xgrid, 1, 1)
#             delta_y = delta_y[np.newaxis, :, np.newaxis]     # Shape: (1, ygrid, 1)
#             delta_z = delta_z[np.newaxis, np.newaxis, :]     # Shape: (1, 1, M)

#             # 3. Compute components A, B, C
#             A = (delta_x * ay1) + (delta_y * ay2) + (delta_z * ay3)  # Shape: (xgrid, ygrid, M)
#             B = (delta_x * az1) + (delta_y * az2) + (delta_z * az3)  # Shape: (xgrid, ygrid, M)
#             C = A + 1j * B  # Shape: (xgrid, ygrid, M)

#             # 5. Compute Term1
#             C1 = a0d / (np.sqrt(2) * 2 * np.pi)
#             Term1 = C1 * np.angle(C)  # Shape: (xgrid, ygrid, M)

#             # 6. Compute Term2
#             Numerator = B * A
#             Denominator = A ** 2 + B ** 2
#             Fraction = Numerator / Denominator
#             Term2 = (C1 / 2.0) * Fraction / (2 * (1 - v))  # Shape: (xgrid, ygrid, M)

#             # 7. Compute self.u
#             self.u = Term1 + Term2  # Shape: (xgrid, ygrid, M)

            # for i in range(self.xgrid):
            #     for j in range(self.ygrid):
            #         for k in range(self.M):
            #             self.u[i,j,k]= 1*(a0d/np.sqrt(2)/2/np.pi*(np.angle(((self.xx[i]-x0d)*az1+(self.yy[j]-y0d)*az2+(z[k]-z0d)*az3)*1j+((self.xx[i]-x0d)*ay1+(self.yy[j]-y0d)*ay2+(z[k]-z0d)*ay3)))+a0d/np.sqrt(2)/2/np.pi/2*(((self.xx[i]-x0d)*az1+(self.yy[j]-y0d)*az2+(z[k]-z0d)*az3)*((self.xx[i]-x0d)*ay1+(self.yy[j]-y0d)*ay2+(z[k]-z0d)*ay3)/(((self.xx[i]-x0d)*ay1+(self.yy[j]-y0d)*ay2+(z[k]-z0d)*ay3)**2+((self.xx[i]-x0d)*az1+(self.yy[j]-y0d)*az2+(z[k]-z0d)*az3)**2)/(2*(1-v))))          
                        
                        
                        
        if self.deformation_model == 'ScrewDislocation': 
            
            #Silicon b=[110],h=[110], tau=[110], n=[-1,1,1],ix=[1,1,0], jy=[-1,1,-2],kz=[-1,1,1]
            z0 = np.abs(self.x00-self.HH*0) / np.tan(self.alpha) # approximate positon of reflection center in z-coordinate 
            z1 = self.Z * np.arange(1, self.M+1) - z0 # z position vector
            z=np.linspace(0,self.Z*self.M,self.M)-self.Z*self.M/2
            x0d=self.config['x0d']*self.convr*1e-6
            y0d=self.config['y0d']*self.convr*1e-6
            phid=self.config['phid']
            slope=self.config['slope']
            z0d=self.config['z0d']*self.convr*1e-6
            a0d=self.a0*self.convr
            burvect_x=self.config['burgers_vector_x']*self.convr*1e-10
            burvect_s=self.config['burgers_vector_screw']*self.convr*1e-10
            v=self.config['vSi']
            
            ay1=  0
            ay2= -1
            ay3= 0
           
            az1= 0
            az2= 0
            az3= 1
            
            # 1. Compute deltas
            delta_x = self.xx - x0d  # Shape: (xgrid,)
            delta_y = self.yy - y0d  # Shape: (ygrid,)
            delta_z = z - z0d        # Shape: (M,)

            # 2. Reshape for broadcasting
            delta_x = delta_x[:, np.newaxis, np.newaxis]  # Shape: (xgrid, 1, 1)
            delta_y = delta_y[np.newaxis, :, np.newaxis]  # Shape: (1, ygrid, 1)
            delta_z = delta_z[np.newaxis, np.newaxis, :]  # Shape: (1, 1, M)

            # 3. Compute real and imaginary parts
            Re = (delta_x * az1) + (delta_y * az2) + (delta_z * az3)
            Im = (delta_x * ay1) + (delta_y * ay2) + (delta_z * ay3)

            # 4. Form complex number
            C = Re + 1j * Im

            # 5. Compute angles
            angles = np.angle(C)

            # 6. Compute the constant factor
            C1 = a0d / (np.sqrt(2) * 2 * np.pi)

            # 7. Compute self.u
            self.u = C1 * angles


                                                       
        if self.deformation_model == 'ThermalBump':
            
            name = self.config['uz_filename']
            self.u = np.load('/pscratch/sd/k/krzywins/CrystalBPM6/crystal-fft-bpm/examples/'+ name) * self.convr
       

        if self.deformation_model == 'None':
            print('No deformation model was selected')
        
        self.qprint('Congigured deformation model')
        
        
    def run3D(self):
        
        start_msg = 'Splitting recipe: ' + self.method
        self.qprint(start_msg)
        
        xbpm = XBPM.XBPM(self)
        
        U1i = 0.0 * xbpm.E_in.copy()
        U2i = 1.0 * xbpm.E_in.copy()
        
        U1f, U2f = xbpm.propagate_with_split_operator_BPM(self, U1i, U2i)
        
        if self.asymm_angle <= 0.0:
            Erefl=np.sum(((self.Xx)<(self.xs-self.HH))*np.abs(U1f)**2)
        else:
            Erefl=np.sum(np.abs(U1f)**2)
            
        
        En0=np.sum(np.abs(xbpm.E_in)**2)
        
        Reflectivity = Erefl / En0 
        Transmission = np.sum(np.abs(U2f)**2.0) / En0
        
        PhaseRefl = np.angle(np.sum(U1f))
        PhaseTrans = np.angle(np.sum(U2f))
        
        self.U1f = U1f
        self.U2f = U2f
        #self.U1f = U1f * np.exp(1j * (np.sin(self.alpha) - self.k0) * self.Xx)
        #self.U2f = U2f * np.exp(-1j *(np.sin(self.alpha) - self.k0) * self.Xx)
        
        self.Reflectivity = Reflectivity
        self.Transmission = Transmission
        self.PhaseRefl = PhaseRefl 
        self.PhaseTrans = PhaseTrans

        
        if self.store_fields:
            self.U1_field = xbpm.U1_store
            self.U2_field = xbpm.U2_store
        
        
        print('Photon energy (omega): ', self.omega, '; Reflectivity: ', Reflectivity, '; Transmission: ', Transmission)
        
        
    def Gronkowski_Chukowski_geometry(self, ay1, ay2, ay3, az1, az2, az3):
        

        x0d=self.config['x0d'] * self.convr * 1e-6
        y0d=self.config['y0d'] * self.convr * 1e-6
        z0d=self.config['z0d'] * self.convr * 1e-6
        a0d=self.a0 * self.convr
        v=self.config['vSi']

        
        z=np.linspace(0,self.Z*self.M,self.M)-self.Z*self.M/2


            
        # 1. Compute deltas
        delta_x = self.xx - x0d  # Shape: (xgrid,)
        delta_y = self.yy - y0d  # Shape: (ygrid,)
        delta_z = z - z0d        # Shape: (M,)

        # 2. Reshape for broadcasting
        delta_x = delta_x[:, np.newaxis, np.newaxis]     # Shape: (xgrid, 1, 1)
        delta_y = delta_y[np.newaxis, :, np.newaxis]     # Shape: (1, ygrid, 1)
        delta_z = delta_z[np.newaxis, np.newaxis, :]     # Shape: (1, 1, M)

        # 3. Compute components A, B, C
        A = (delta_x * ay1) + (delta_y * ay2) + (delta_z * ay3)  # Shape: (xgrid, ygrid, M)
        B = (delta_x * az1) + (delta_y * az2) + (delta_z * az3)  # Shape: (xgrid, ygrid, M)
        C = A + 1j * B  # Complex array

        # 5. Compute Term1
        C1 = a0d / (np.sqrt(2) * 2 * np.pi)
        Term1 = C1 * np.angle(C)  # Shape: (xgrid, ygrid, M)

        # 6. Compute Term2
        Numerator = B * A
        Denominator = A ** 2 + B ** 2
        Fraction = Numerator / Denominator

        Term2 = C1 / 2.0 * Fraction / (2 * (1 - v))  # Shape: (xgrid, ygrid, M)

        # 7. Compute self.u
        self.u = (Term1 + Term2)*1  # Shape: (xgrid, ygrid, M)
        
        
#     def test(self):
        
#         start_msg = 'Splitting recipe: ' + self.method
#         self.qprint(start_msg)
        
#         xbpm = XBPM.XBPM(self)
#         ksak=ksih1_m_select(self)
#         self.ksak=ksak
#         print('ksih1_m_select',self.ksak)
        
