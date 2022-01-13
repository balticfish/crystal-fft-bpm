import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.interpolate import RegularGridInterpolator
import time
import scipy.fft as sp_fft


class XCrTools:
    
    def __init__(self, XCr):
        
        self.nthread_fft = XCr.nthread_fft
        XCr.qprint('Initialized tools...')
        
        
    def Gaussian_2D(self, XCr, alpha=None):
        '''
        Def. of Gaussian beam with  angle of incidence alpha, waist radius om0,position from the source zX and shifted w/r to origin by x00 and with the fast oscilating component exp(1j*k0*x) removed
        '''
        if (alpha is None):
            alpha = XCr.alpha
        
        return XCr.E0 * 1.0 / (1.0 + 1j * XCr.zX / XCr.zR) * np.exp(-(((((XCr.Xx - XCr.x00)**2.0 + XCr.Yy**2.0) / ((XCr.om0**2.0) * (1.0 + 1j * XCr.zX / XCr.zR)))))) * np.exp(1j * (np.sin(alpha) - XCr.k0) * XCr.Xx)
    
#     def Gaussian_pulse_3D_with_q(self, XCr, k=None):
#         """
#         Generate a complex three-dimensional spatio-temporal Gaussian field amplitude profile with in terms of complex beam parameter q

#         Parameters
#         ----------
  
#         Returns
#         -------
#         np.ndarray

#         """

#         qx = 1j * XCr.zR
#         qy = 1j * XCr.zR

#         ux = 1.0 / np.sqrt(qx) * np.exp(-1j * XCr.K0 * (XCr.Xx - XCr.x00)**2 / 2.0 / qx)
#         uy = 1.0 / np.sqrt(qy) * np.exp(-1j * XCr.K0 * XCr.Yy**2 / 2.0 / qy)
#         ut = 1.0 / (np.sqrt(2.0 * np.pi) * X.sigma_t) * np.exp(-(X.t_mesh - X.t0)**2 / 2.0 / X.sigma_t**2)

#         eta = 2.0 * k * X.zR * X.sigma_t / np.sqrt(np.pi)

#         return np.sqrt(eta) * np.sqrt(X.N_pump) * ux * uy * ut * np.exp(1j * (np.sin(alpha) - XCr.k0) * XCr.Xx)
    
    
    def Pseudo_SASE_3D(self, XCr, tau, sigma_coh, Nmodes, Nslice):
        '''
        Def. of SASE beam with angle of incidence alpha, waist radius om0,position from the source zX and shifted w/r to origin by x00 with the fast oscilating component exp(1j*k0*x) removed, and with pseudo-random temporal structure.
        '''
        qx = 1j * XCr.zR
        qy = 1j * XCr.zR
        
        ux = 1.0 / np.sqrt(qx) * np.exp(-1j * XCr.K0 * (XCr.Xx - XCr.x00)**2 / 2.0 / qx)
        uy = 1.0 / np.sqrt(qy) * np.exp(-1j * XCr.K0 * XCr.Yy**2 / 2.0 / qy)
        
        uxuy = ux * uy * np.exp(1j * (np.sin(XCr.alphaB) - XCr.k0) * XCr.Xx)
        
        t0 = np.random.uniform(-tau/2.0, tau/2.0, Nmodes)
        t = np.linspace(-tau/2.0, tau/2.0, Nslice)
        
        omega0 = 2.0 * np.pi * XCr.c / XCr.lam0

        ut = 1.0 / (np.sqrt(2.0 * np.pi) * sigma_coh) * np.exp(-(t - t0[0])**2 / 2.0 / sigma_coh**2 - 1j * omega0 * (t - t0[0]))
                
        for i in range(1, Nmodes):
            ut += 1.0 / (np.sqrt(2.0 * np.pi) * sigma_coh) * np.exp(-(t - t0[i])**2 / 2.0 / sigma_coh**2 - 1j * omega0 * (t - t0[i]))
            
        ut /= (Nmodes * 1.0)
                            
        return np.einsum('t, xy->txy', ut, uxuy)
    
    
    def fft2(self, fftw_in):
        return sp_fft.fftn(fftw_in, axes=(0,1), workers=self.nthread_fft, overwrite_x=False)

    def ifft2(self, ifftw_in):
        return sp_fft.ifftn(ifftw_in, axes=(0,1), workers=self.nthread_fft, overwrite_x=False) 
    
    def intensity(self, U):
        return np.sum(np.abs(U)**2.0)
    
    def field_from_file_genesis2_DFL(self, XCr, crop_t = 1.0, crop_x = 1.0):
        """
        fname: string
            filename
        nx: int
            grid size in x and y. Same as GenesisV2 'ncar'

        Returns 3d numpy.array with indices as:
            [t, x, y]

        Adopted from lume-genesis package; see https://github.com/slaclab/lume-genesis
        """

        dat = np.fromfile(XCr.fname_GenV2, dtype=np.complex).astype(np.complex)
        npoints = dat.shape[0] 
        nx = XCr.ncar_GenV2
        ny = nx
        nt =  npoints / ny / nx
        assert (nt % 1 == 0), f'Confused shape {nt} {nx} {ny}' 
        nt = int(nt)    
        dat = np.reshape(dat, (nt, nx, ny))  
        dat = self.crop_3d_wavefront(dat, [crop_t, crop_x, crop_x])
        ntc, nxc, nyc = np.shape(dat)

        tmax = ntc * XCr.lam * XCr.zsep_GenV2 / XCr.c 
        print('DFL tmax: ', tmax, 'fs')

        import matplotlib.pyplot as plt
        plt.title('loaded field temporal profile')
        plt.plot(np.linspace(0, tmax, ntc), np.sum(np.real(dat * np.conj(dat)), axis=(1,2)))
        plt.xlabel('Time (fs)')
        plt.ylabel('arb.')

#     old_domain = (np.linspace(0, tmax, ntc), np.linspace(-XCr.dgrid_GenV2, XCr.dgrid_GenV2, nxc), np.linspace(-XCr.dgrid_GenV2, XCr.dgrid_GenV2, nyc))
#     new_mesh = (X.t_mesh, X.x_mesh, X.y_mesh)
    
#     dat_intrp = interpolate_wavefront_3D(dat, old_domain, new_mesh)
    
  #  fudge = np.sum(np.real(dat_intrp * np.conj(dat_intrp))) * X.dt * X.dx * X.dy / X.N_pump

        return dat#_intrp #/ np.sqrt(fudge)


    def crop_3d_wavefront(self, wavefront3D, cropping_factors=[3, 3, 3]):

        mt0, mx0, my0 = np.shape(wavefront3D)
        mt = mt0 / cropping_factors[0]
        mx = mx0 / cropping_factors[1]
        my = my0 / cropping_factors[2]

        print('Cropped pump field to a new shape (t, x, y): ', int(mt), int(mx), int(my))

        return wavefront3D[int(mt0/2 - mt/2):int(mt0/2 + mt/2), int(mx0/2 - mx/2):int(mx0/2 + mx/2), int(my0/2 - my/2):int(my0/2 + my/2)]


    def interpolate_wavefront_3D(self, wavefront3D, old_domain, new_mesh):

        wvf3D_real = RegularGridInterpolator(old_domain, np.real(wavefront3D), method='linear', bounds_error=False, fill_value=None)
        wvf3D_imag = RegularGridInterpolator(old_domain, np.imag(wavefront3D), method='linear', bounds_error=False, fill_value=None)

        wvf3D_intrp = (wvf3D_real(new_mesh) + 1j * wvf3D_imag(new_mesh))

        return wvf3D_intrp
    
    
    def my_pad(self, wavefront, shape):
        """
        Pad an array with complex zero elements.

        Parameters
        ----------
        wavefront: np.ndarray
            The array to pad
        shape: array_like
            Number of values padded to the edges of each axis

        Returns
        -------
        np.ndarray

        """

        return np.pad(wavefront, shape, mode='constant', constant_values=(0.0 + 1j*0.0, 0.0 + 1j*0.0))
    
    
    def nd_kspace(self, coeffs=(), sizes=(), pads=(), steps=()):

        domains = [coeff * np.fft.fftfreq(n + 2 * pad, step)  for coeff, n, pad, step in zip(coeffs, sizes, pads, steps)]
        meshes = np.meshgrid(*domains, indexing='ij')
        step_sizes = [domain[1] - domain[0] for domain in domains]

        return domains, meshes, step_sizes  