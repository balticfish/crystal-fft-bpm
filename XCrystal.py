import numpy as np
from scipy.special import jv
import scipy.constants as sp_const
import yaml
import XBPM as XBPM

class XCrystal:

    def __init__(self, YAML, omega=None, E0=None):
        
        self.config = yaml.load(open(YAML), Loader=yaml.FullLoader)
        
        self.omega0 = self.config['omega0']
        
        if (omega is None):
            self.omega = self.omega0
        else:
            self.omega = omega 
            
        self.lam0 = 12398.0 / self.omega0  * 1.0e-10
        self.convr0 = 2.0 * np.pi / self.lam0 # 'convr0' to convert meters to dimensionless units

        self.lam = 12398.0 / self.omega  * 1.0e-10
        self.K0 = 2.0 * np.pi / self.lam #modululs of k vector
        self.K00 = 2.0 * np.pi / self.lam0 #modululs of k vector
        self.convr = self.K0 # 'convr' to convert meters to dimensionless units
          
        self.c = sp_const.c
        self.hbar = sp_const.hbar / sp_const.e 

        self.a0 = self.config['a0'] * 1.0e-10
        self.Miller_h = self.config['Miller_h']
        self.Miller_k = self.config['Miller_k']
        self.Miller_l = self.config['Miller_l']
        
        self.d = np.sqrt(self.a0**2 / (self.Miller_h**2 + self.Miller_k**2 + self.Miller_l**2)) * self.convr #interplanar spacing
        #self.d = np.sqrt(self.a0**2 / (self.Miller_h**2 + self.Miller_k**2 + self.Miller_l**2)) * self.convr0 #interplanar spacing
        self.dm = self.d / self.convr #interplanar spacing [m]
        
        self.alphaB = np.arcsin(self.lam0 / self.dm / 2.0)

        self.k0 = np.pi / self.d # half of reciprocal vector,  reciprocal vector =  2*k0
        self.n_d = 1.0 - (self.config['delta'] -1j * self.config['beta'])
        self.xrh = self.config['xrh']
        self.xih = self.config['xih']
        self.ele_susceptH = (self.xrh - 1j*self.xih)
        self.eps = self.n_d**2.0 # average electric susceptibility 
        self.epsxh0 = self.eps - 1.0 # difference of average electric susceptibility w.r. to vacuum
        self.cosa = np.sqrt(1.0 - self.k0**2.0)
        
        self.Zstep_factor = self.config['Zstep_factor']
        self.Z = np.abs(self.Zstep_factor * np.pi/np.real(self.epsxh0))  #  Z - step in z
        self.zsep = self.config['zsep']
        self.nthread_fft = self.config['nthread_fft']
        
        self.HH = self.config['thickness'] * 1.0e-6 / 2.0 * self.convr
        self.sep1=self.config['separation1'] * 1.0e-6 * self.convr
        self.sep2=self.config['separation2'] * 1.0e-6 * self.convr
        self.method = self.config['method']
        self.quiet = self.config['quiet_mode']
        self.store_fields = self.config['store_fields']

        if (self.quiet == False):
            print('Quiet mode disabled. I will talk a lot...')
        
        
        self.xxmax = self.config['xxmax'] * 1.0e-6 * self.convr # grid size in x
        self.yymax = self.config['yymax'] * 1.0e-6 * self.convr # grid size in y 

        

        self.xgrid = self.config['xgrid']
        self.ygrid = self.config['ygrid']
        self.tpad = self.config['tpad']
        
       # self.nxarr = 2.0 * self.xxmax / self.dxx
        #self.nyarr = 2.0 * self.yymax / self.dyy


        self.xx =  np.linspace(-self.xxmax, self.xxmax, self.xgrid)
        self.yy =  np.linspace(-self.yymax, self.yymax, self.ygrid)
        
        self.Yy, self.Xx = np.meshgrid(self.yy, self.xx) # x,y mesh/grid
        
        # reciprocal space (angular spectrum)
        self.dkx = 2.0*np.pi/max(self.xx)/2.0  # grid resolution,k vector in x  
        self.kkx = self.dkx*(np.arange(1,len(self.xx)+1) - 0.5*len(self.xx))#  k vector in x   
        self.dky = 2.0*np.pi/max(self.yy)/2.0 # grid resolution,k vector in y 
        self.kky = self.dky*(np.arange(1,len(self.yy)+1) - 0.5*len(self.yy)) #k vector in y
        self.Ky, self.Kx = np.meshgrid(self.kky, self.kkx) # angular spectrum mesh/grid
        
        
        self.qprint('Congigured grid parameters')
        self.beam = self.config['beam']
        
        if self.beam=='Gaussian':
            self.waist = self.config['waist']
            self.om0  = self.waist * 1e-6 * self.convr   # electric field radius at the waist
            self.omZ = self.waist * 1e-6 * self.convr   # electric field radius at the sample
            self.zR  = self.K0 * (self.om0**2.0)/2.0     # Rayleigh parameter in the internal units (lamda =2*np.pi)
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
        
        if self.quiet:
            pass
        else:
            print(str)
            
    def configure(self, Delta_alpha, field=None):
        
        self.Delta_alpha = Delta_alpha
        self.alpha = self.alphaB + self.Delta_alpha
        self.alphaDeg = self.alpha * 180.0 / np.pi
        
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
        
        if self.deformation_model == 'Sinusoidal':
            zee = np.arange(1, 1e4+1) * 1e-7 - 0.5e-3 # z-mesh [m]
            lam_z = self.config['def_period'] * 1.0e-6 
            kz = 2.0 * np.pi / lam_z # undulation period [m]
            amp = self.config['def_amplitude'] # undulation amplidude
            uee = amp * np.cos(zee * kz) #undulation function [m]
            ze = zee * self.convr  # undulations in dimensionless units 
            ue = uee * self.convr  # undulations in dimensionless units
            z0 = np.abs(self.x00-self.HH) / np.tan(self.alpha) # approximate positon of reflection center in z-coordinate 
            z1 = self.Z * np.arange(1, self.M+1) - z0 # z position vector
            u_z = np.interp(z1,ze,ue) # displacement of crystal planes at z1
            self.u = np.zeros((self.xgrid,self.ygrid,self.M))
       
            for i in range(self.xgrid):
                for j in range(self.ygrid):
                    for k in range(self.M):
                        self.u[i,j,k]=u_z[k]
            
        if self.deformation_model == 'ConstStrainGrad':
            B1 = self.config['def_B']/self.convr
            u_x = B1*(self.xx+self.HH)**2
            self.u = np.zeros((self.xgrid,self.ygrid,self.M))
       
            for i in range(self.xgrid):
                for j in range(self.ygrid):
                    for k in range(self.M):
                        self.u[i,j,k]=u_x[i]

        if self.deformation_model == 'ThermalBump':
            name = self.config['uz_filename']
            self.u = np.load('/global/cscratch1/sd/krzywins/CRYSTALBPMExpl/crystal-fft-bpm/examples/'+ name)*self.convr*1
       
                    
       
            
        if self.deformation_model == 'None':
            self.u = np.zeros((self.xgrid,self.ygrid,self.M))
        
        self.qprint('Congigured deformation model')
        
        
    def run3D(self):
        
        start_msg = 'Splitting recipe: ' + self.method
        self.qprint(start_msg)
        
        xbpm = XBPM.XBPM(self)
        
        U1i = 0.0 * xbpm.E_in
        U2i = 1.0 * xbpm.E_in
        
        U1f, U2f = xbpm.propagate_with_split_operator_BPM(self, U1i, U2i)
        
        Reflectivity = np.sum(np.abs(U1f)**2.0) / np.sum(np.abs(xbpm.E_in)**2)
        Transmission = np.sum(np.abs(U2f)**2.0) / np.sum(np.abs(xbpm.E_in)**2)
        PhaseRefl = np.angle(np.sum(U1f))
        PhaseTrans = np.angle(np.sum(U2f))
        self.U1f = U1f
        self.U2f = U2f
        #self.U1f = U1f * np.exp(1j * (np.sin(self.alpha) - self.k0) * self.Xx)
        #self.U2f = U2f * np.exp(-1j *1*(np.sin(self.alpha) - self.k0) * self.Xx)
        
        self.Reflectivity = Reflectivity
        self.Transmission = Transmission
        self.PhaseRefl = PhaseRefl 
        self.PhaseTrans = PhaseTrans

        
        if self.store_fields:
            self.U1_field = xbpm.U1_store
            self.U2_field = xbpm.U2_store
        
        
        print('Delta theta: ', self.Delta_alpha, '; Reflectivity: ', Reflectivity, '; Transmission: ', Transmission, 'PhaseRefl', PhaseRefl )
        
        
