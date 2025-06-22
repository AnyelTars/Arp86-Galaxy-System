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
    def __init__(self,path_directory,cube_name,optical_name,mask_name=None):
        ''' Initialize instance attributes '''

        self.path_directory = path_directory
        self.cube_name = cube_name
        self.mask_name = mask_name
        self.optical_name = optical_name
    
    def read_files(self,type="cube"):
        ''' Read cube data (fits file)
        Inputs: cube (it is the type of the fits data cube, it is a string and could be either: cube, mask, optical)
        
        '''
        #data cube
        if type == "cube":
            hdul = fits.open(self.path_directory+self.cube_name)
            header = hdul[0].header
            data = hdul[0].data
            return data,header
        elif type == "mask":
            hdul = fits.open(self.path_directory+self.mask_name)
            header = hdul[0].header
            data = hdul[0].data
            return data,header
        elif type == "optical":
            hdul = fits.open(self.path_directory+self.optical_name)
            header = hdul[0].header
            data = hdul[0].data
            return data,header
        else: 
            print('Error: type value mustr be a string (cube,mask or optical)')

    def compute_moments(self):
        '''  
        
        '''
        data,header = self.read_files(type='cube') 
        if self.mask_name is not None:
            data_mask,header_mask = self.read_files(type="mask")
            masked_data = np.where(data_mask,data,0)
            # integrated intensity (moment 0)
            mom0_data = np.nansum(masked_data,axis=0)*abs(header['CDELT3'])/1000

            # velocity field (moment 1)
            d_1_av=np.nansum(masked_data[:,:,:],axis=1)
            d_av=np.nansum(d_1_av[:,:],axis=1)
            spectra=d_av[:-3]
            x_axis=(np.arange(0,len(spectra))*header['CDELT3']+(header['CRVAL3']-header['CRPIX3']*header['CDELT3']))/1000
            d_for_mom1=masked_data.copy()
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
            mom0_data = np.nansum(data,axis=0)*abs(header['CDELT3'])/1000

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
            d_for_mom2=data.copy()
            for i in range(len(x_axis)):
                d_for_mom2[i,:,:]=(x_axis[i]-mom1_data)**2 * d_for_mom2[i,:,:]
            mom2_data = np.sqrt(np.nansum(d_for_mom2, axis=0)*(abs(header['CDELT3'])/(mom0_data*1000)))

        return mom0_data, mom1_data, mom2_data, header
        
    def save_data(self,name_galaxy):
        ''' 
        Save cube data for intensity, velocity field, and velocity dispersion (moment 1, moment 2, moment 3).
        -h_1 is the header of the original data cube
        -moment is the name of the  map, this will be used to give the name for the fits file
        -data_mom is the data of the moment map  

        Imputs: name_galaxy (String providing the name of the galaxy)
        '''
        # get moments data
        self.moment0_data,self.moment1_data,self.moment2_data,header = self.compute_moments()
        # write header for the new data cubes
        wcs_new = WCS(naxis=2)
        wcs_new.wcs.cdelt = [header['CDELT1'], header['CDELT2']]
        wcs_new.wcs.ctype = [header['CTYPE1'], header['CTYPE2']]
        wcs_new.wcs.crval = [header['CRVAL1'], header['CRVAL2']]
        wcs_new.wcs.crpix = [header['CRPIX1'], header['CRPIX2']]
        header_new = wcs_new.to_header()        

        # add the data and the header to an object that can be written into a file 
        # moment 0 
        self.hdul_new0 = fits.PrimaryHDU(self.moment0_data, header=header_new) 
        self.hdul_new0.writeto(self.path_directory+'{}_.fits'.format(f'{name_galaxy}_moment0'),overwrite=True)
        #moment 1
        self.hdul_new1 = fits.PrimaryHDU(self.moment1_data, header=header_new) 
        self.hdul_new1.writeto(self.path_directory+'{}_.fits'.format(f'{name_galaxy}_moment1'),overwrite=True)
        
        #moment 2
        self.hdul_new2 = fits.PrimaryHDU(self.moment2_data, header=header_new) 
        self.hdul_new2.writeto(self.path_directory+'{}_.fits'.format(f'{name_galaxy}_moment2'),overwrite=True)
        print(f'data cubes sucessfully saved in {self.path_directory} as: ')
        print(f'{'{}_.fits'.format(f'{name_galaxy}_moment0')}')
        print(f'{'{}_.fits'.format(f'{name_galaxy}_moment1')}')
        print(f'{'{}_.fits'.format(f'{name_galaxy}_moment2')}')

        self.name_mom0 = '{}_.fits'.format(f'{name_galaxy}_moment0')
        self.name_mom1 = '{}_.fits'.format(f'{name_galaxy}_moment1')
        self.name_mom2 = '{}_.fits'.format(f'{name_galaxy}_moment2')
     
    def plot_fields(self):
        '''
        This methods plots the integrated intensity, velocity field, and velocity dispersion
        '''
        # moment 0 
        header_mom0 = fits.open(self.path_directory+self.name_mom0)[0].header
  
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(1,1,1,projection = (WCS(header_mom0)))
        cb = plt.imshow(self.moment0_data,cmap='magma')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        cbar = fig.colorbar(cb)
        cbar.minorticks_on()
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel(r'Jy beam${-1}$ km${-1}$',fontsize=20)
        plt.show()

        # moment 1
        header_mom1 = fits.open(self.path_directory+self.name_mom1)[0].header
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(1,1,1,projection = (WCS(header_mom1)))
        cb = plt.imshow(self.moment1_data,cmap='RdBu_r')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        cbar = fig.colorbar(cb)
        cbar.minorticks_on()
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel(r'Km s${-1}$',fontsize=20)
        plt.show()

        # moment 3
        header_mom2 = fits.open(self.path_directory+self.name_mom2)[0].header
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(1,1,1,projection = (WCS(header_mom2)))
        cb = plt.imshow(self.moment2_data,cmap='viridis')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        cbar = fig.colorbar(cb)
        cbar.minorticks_on()
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel(r'Km s${-1}$',fontsize=20)
        plt.show()

    def plot_overlays(self):
        
        # prepare the data 
        hdul_mom0 = fits.open(self.path_directory+self.name_mom0)
        hdul_mom1 = fits.open(self.path_directory+self.name_mom1)
        hdul_mom2 = fits.open(self.path_directory+self.name_mom2)
        header_mom0 = hdul_mom0[0].header
        header_mom1 = hdul_mom1[0].header
        header_mom2 = hdul_mom2[0].header
        
        #
    
        data_optical,header_optical = self.read_files(type="optical")
        # moment 0
        hi_reprojected_0,footprint_0 = reproject_interp(hdul_mom0,header_optical)
        rms_mom0 = np.sqrt(np.nanmean(hi_reprojected_0**2))
        m0max = np.nanmax(hi_reprojected_0)
        m0min = np.nanmin(hi_reprojected_0) 
        steps_0 = (m0max-m0min)*5/100
        #moment 1 
        hi_reprojected_1,footprint_1 = reproject_interp(hdul_mom1,header_optical)
        m1max = np.nanmax(hi_reprojected_1)
        m1min = np.nanmin(hi_reprojected_1) 
        steps_1 = (m1max-m1min)*5/100

        #moment 2
        hi_reprojected_2,footprint_2 = reproject_interp(hdul_mom2,header_optical)
        m2max = int(np.ceil(np.nanmax(hi_reprojected_2)))
        m2min = int(np.nanmin(hi_reprojected_2))
        steps_2 = (m2max-m2min)*5/100

        # plot overlay integrated intensity and optical image
        fig=plt.figure(figsize=(9,8))
        ax = fig.add_subplot(1,1,1, projection=WCS(header_mom0))
        cb=ax.imshow(data_optical,cmap='gist_heat',vmax=np.percentile(data_optical,99.8))
        #ax.contour(hi_reprojected_0,levels=np.arange(0,2.8,0.2)*rms_mom0,cmap='Blues_r')
        ax.contour(hi_reprojected_0,levels=np.arange(m0min,m0max,steps_0),cmap='Blues_r')
        ax.set_xlabel('RA', size=18,family='serif')
        ax.set_ylabel('Dec', size=18,family='serif')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()

        # plot overlay velocity field and optical image
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(1,1,1, projection=WCS(header_mom1))
        ax.imshow(data_optical, cmap='gist_gray',vmax=np.percentile(data_optical, 99.8))
        ax.contour(hi_reprojected_1, levels=np.arange(m1min,m1max,steps_1),cmap='RdBu_r')
        ax.set_xlabel('RA', size=18,family='serif')
        ax.set_ylabel('Dec', size=18,family='serif')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()

        # plot overlay velocity dispersion and optical image
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(1,1,1, projection=WCS(header_mom2))
        ax.imshow(data_optical, cmap='gist_heat',vmax=np.percentile(data_optical, 99.8))
        ax.contour(hi_reprojected_2, levels=np.arange(m2min, m2max,steps_2),cmap='viridis')
        ax.set_xlabel('RA', size=18,family='serif')
        ax.set_ylabel('Dec', size=18,family='serif')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()