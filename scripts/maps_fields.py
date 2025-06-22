import  numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from reproject import reproject_interp
from astroquery.skyview import SkyView
from astroquery.vizier import Vizier
from spectral_cube import SpectralCube
import glob
from PIL import Image
import os
from IPython import display
import astropy.wcs.utils as util
from matplotlib.patches import Circle
class Galaxy_maps:
    ''' This class contains all the methods to compute the integrated intensity, velocity field and the velocity dispersion of the HI content of the galaxy'''
    
    # Class variables

    # Initializer (constructor)
    def __init__(self,path_directory,cube_name, mask_name=None):
        ''' Initialize instance attributes '''

        self.path_directory = path_directory
        self.cube_name = cube_name
        self.mask_name = mask_name
    
    def read_files(self):
        ''' Read cube data (fits file)'''
        #data cube
        hdul = fits.open(self.path_directory+self.cube_name)
        header = hdul[0].header
        data = hdul[0].data
        return data,header

    
    def compute_moments(self):
        '''  
        
        '''
        data,header = self.read_files() 
        if self.mask_name is not None:
            data_mask,header_mask = self.read_files()
            masked_data = np.where(data_mask,data,0)
            # integrated intensity (moment 0)
            mom0_data = np.nansum(masked_data,axis=0)*abs(self.header['CDELT3'])/1000

            # velocity field (moment 1)
            d_1_av=np.nansum(data[:,:,:],axis=1)
            d_av=np.nansum(d_1_av[:,:],axis=1)
            spectra=d_av[:-3]
            x_axis=(np.arange(0,len(spectra))*header['CDELT3']+(header['CRVAL3']-header['CRPIX3']*header['CDELT3']))/1000
            d_for_mom1=data.copy()
            for i in range(len(x_axis)):
                d_for_mom1[i,:,:] = x_axis[i]*d_for_mom1[i,:,:]
            mom1_data=np.nansum(d_for_mom1,axis=0)*abs(header['CDELT3'])/(mom0_data*1000)
            
            # integrated intensity (moment 2)
            d_for_mom2=masked_data.copy()
            for i in range(len(x_axis)):
                d_for_mom2[i,:,:]=(x_axis[i]-mom1_data)**2 * d_for_mom2[i,:,:]
            mom2_data = np.sqrt(np.nansum(d_for_mom2, axis=0)*(abs(header['CDELT3'])/(mom0_data*1000)))
        else: 
            # integrated intensity (moment 0)
            mom0_data = np.nansum(data,axis=0)*abs(self.header['CDELT3'])/1000

            # velocity field (moment 1)
            d_1_av=np.nansum(data[:,:,:],axis=1)
            d_av=np.nansum(d_1_av[:,:],axis=1)
            spectra=d_av[:-3]
            x_axis=(np.arange(0,len(spectra))*header['CDELT3']+(header['CRVAL3']-header['CRPIX3']*header['CDELT3']))/1000
            d_for_mom1=data.copy()
            for i in range(len(x_axis)):
                d_for_mom1[i,:,:] = x_axis[i]*d_for_mom1[i,:,:]
            mom1_data=np.nansum(d_for_mom1,axis=0)*abs(header['CDELT3'])/(mom0_data*1000)
            
            # integrated intensity (moment 2)
            d_for_mom2=masked_data.copy()
            for i in range(len(x_axis)):
                d_for_mom2[i,:,:]=(x_axis[i]-mom1_data)**2 * d_for_mom2[i,:,:]
            mom2_data = np.sqrt(np.nansum(d_for_mom2, axis=0)*(abs(header['CDELT3'])/(mom0_data*1000)))

        return mom0_data, mom1_data, mom2_data
        
    def save_data(self,name_galaxy):
        ''' 
        Save cube data for intensity, velocity field, and velocity dispersion (moment 1, moment 2, moment 3).
        -h_1 is the header of the original data cube
        -moment is the name of the  map, this will be used to give the name for the fits file
        -data_mom is the data of the moment map  

        Imputs: name_galaxy (String providing the name of the galaxy)
        '''
        wcs_new = WCS(naxis=2)
        wcs_new.wcs.cdelt = [self.header['CDELT1'],self.header['CDELT2']]
        wcs_new.wcs.ctype = [self.header['CTYPE1'], self.header['CTYPE2']]
        wcs_new.wcs.crval = [self.header['CRVAL1'], self.header['CRVAL2']]
        wcs_new.wcs.crpix = [self.header['CRPIX1'], self.header['CRPIX2']]
        header = wcs_new.to_header()        

        # get moments data
        self.moment0_data,self.moment1_data,self.moment2_data = self.compute_moments()
        # add the data and the header to an object that can be written into a file 
        # moment 0 
        hdul_new0 = fits.PrimaryHDU(self.moment0_data, header=header) 
        hdul_new0.writeto(self.path_directory+'{}_.fits'.format(f'{name_galaxy}_moment0'),overwrite=True)
        #moment 1
        hdul_new1 = fits.PrimaryHDU(self.moment1_data, header=header) 
        hdul_new1.writeto(self.path_directory+'{}_.fits'.format(f'{name_galaxy}_moment1'),overwrite=True)
        
        #moment 2
        hdul_new2 = fits.PrimaryHDU(self.moment2_data, header=header) 
        hdul_new2.writeto(self.path_directory+'{}_.fits'.format(f'{name_galaxy}_moment2'),overwrite=True)
        print(f'data cubes sucessfully saves in {self.path_directory}')
    
    def plot_fields(self):
        # moment 0
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(1,1,1,projection = (WCS(self.header)))
        cb = plt.imshow(self.moment0_data,cmap='magma_r')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        cbar = fig.colorbar(cb)
        cbar.minorticks_on()
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel(r'Km s${-1}$',fontsize=20)
        plt.show()
        
        # moment 1
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(1,1,1,projection = (WCS(self.header)))
        cb = plt.imshow(self.moment1_data,cmap='magma_r')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        cbar = fig.colorbar(cb)
        cbar.minorticks_on()
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel(r'Km s${-1}$',fontsize=20)
        plt.show()
