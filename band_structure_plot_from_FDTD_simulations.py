#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:39:06 2024

@author: winniep01
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
from scipy.constants import h, e
# Constants
hevs = h/e  # Planck's constant in eV·s

# This script assumes that you ran the script 'da.lsf' that saves the time-domain electric fields in the following folder structure:
#   main_folder/
#        monitors/
#           0/          # monitor folder names can be set using the k_point_folders variable
#           0_0500/
#           0_1000/
#           ...
#               1_Ex.txt
#               1_Ey.txt
#               1_Ez.txt
#               2_Ex.txt
#               ...
#  

def band_structure_FDTD(filepath, k_point_folders, k_point_values, ymin, ymax, imshowplot=False, logscale=False, time_monitors=[1,2,3,4,5,6,7,8,9,10], vmin=None, vmax=None, xlabel=None):
    # parameters:
    # filepath:         filepath to main folder containing the monitors folder
    # k_points:         names of subfolders containing the time-domain monitors for each k point
    # k_point_values:   array of normalized k points from 0 to 1. Here you can enter uneven k point spacing to get higher resolution at specific momenta
    # ymin, ymax:       plot ymin and ymax in eV
    # imshowplot:       flag to include plotting, set to false if you just want the band structure data for further processing
    # logscale:         flag to enable log-scale intensity plotting
    # time_monitors:    array to specify the time monitors you want to include in the analysis. Useful to determine the differences in the band structure
    #                   at different locations in the unit cell.
    # vmin, vmax:       min and max intensity for the colorbar. vmin has to be greater than zero for logscale plots
    # xlabel:           label for the x axis. Works with Latex code such as xlabel=['$\Gamma$','$\mathrm{L}$']
    
    intensity_data = [] # list to hold fft intensities
    dispy = []          # list to hold fft frequencies
    
    for k_point in k_point_folders:
        fftavg = 0
        fftlist = []
        for index in time_monitors:
            # Load time-domain electric field data
            datax = np.loadtxt(filepath+k_point+f"{index}_Ex.txt", delimiter="\t", skiprows=0)
            datay = np.loadtxt(filepath+k_point+f"{index}_Ey.txt", delimiter="\t", skiprows=0)
            dataz = np.loadtxt(filepath+k_point+f"{index}_Ez.txt", delimiter="\t", skiprows=0)
        
        
            time = datax[:, 0]  # Time in seconds
            Ex = datax[:, 1]  # Electric field values in V/m
            Ey = datay[:, 1]  # Electric field values in V/m
            Ez = dataz[:, 1]  # Electric field values in V/m
        
            # set parameters for apodization
            apod_width = 0.125;
            apod_center = 0.5;
            
            # Number of points in the time series
            N = len(time)*5
            
            E_tot = Ex + Ey + Ez
            E_tot = E_tot*np.exp( - 0.5*(time-max(time)*apod_center)**2/(apod_width*max(time))**2)
            E_field_fft = fft(E_tot, N)
            fftlist.append(np.abs(E_field_fft)**2)
            
        # perform the FFT and average over all selected monitors for each momentum 
        frequencies = fftfreq(N, (time[1]-time[0]))  # Frequencies in Hz        
        frequencies = frequencies * hevs  # Convert Hz to eV
        fftavg = np.array([sum(np.transpose(fftlist)[k])/len(fftlist) for k in range(len(fftlist[0]))])
        intensity_data.append(fftavg)
        dispy.append(np.array(frequencies))
    
    x_data = k_point_values 
    y_data = dispy
    
    # Interpolate the FFT data because the duration of an FDTD simulation varies depending  
    # on the lifetime of the modes.
    
    # Define a common y-grid for interpolation
    common_y = np.linspace(min(min(y) for y in y_data), max(max(y) for y in y_data), 5000) # 50000
    
    # Create empty array for interpolated intensity data
    grid_intensity = np.zeros((len(x_data), len(common_y)))
    
    for i in range(len(x_data)):
        # Perform 1D interpolation along the y-axis
        f_interp = interp1d(y_data[i], intensity_data[i], kind='linear', fill_value=0, bounds_error=False)
        
        # Interpolate intensity values along the common y-grid
        grid_intensity[i, :] = f_interp(common_y)
    
    
    # Now we need to interpolate the x positions using nearest-neighbor interpolation
    # Define a new uniform x-grid for imshow
    num_x_points = 50 #100  # You can adjust this number to control the interpolation resolution of the x-axis
    common_x = np.linspace(0, 1, num_x_points)  # Uniform x positions between 0 and 1
    
    # Nearest-neighbor interpolation along the x-axis
    f_x_interp = interp1d(x_data, grid_intensity, axis=0, kind='nearest', fill_value="extrapolate")
    
    # Get the interpolated intensity values on the common x-grid
    interpolated_grid_intensity = f_x_interp(common_x)
        
    if imshowplot:    
        plt.figure(figsize=(4,8))
        xmin = common_x.min()
        xmax = common_x.max()
            
        if logscale:
            # Plot band structure image using imshow
            plt.imshow(interpolated_grid_intensity.T, extent=[xmin, xmax, common_y.min(), common_y.max()],
                       aspect='auto', origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
            
                
            # Set labels and colorbar
            plt.colorbar(label='$\mathrm{intensity \ (arb. \ unit)}$')
        else:
            # Plot band structure image using imshow
            plt.imshow(interpolated_grid_intensity.T, extent=[xmin, xmax, common_y.min(), common_y.max()],
                       aspect='auto', origin='lower', vmin=vmin, vmax=vmax) 
            
            # Set labels and colorbar
            plt.colorbar(label='$\mathrm{intensity \ (arb. \ unit)}$')

        plt.xticks([0,1], [xlabel[0], xlabel[1]])

        plt.ylim([ymin, ymax])
        plt.xlim([0.0, 1])
        plt.ylabel('$\mathrm{energy \ (eV)}$')
        plt.tight_layout()
        plt.show()
    
    return interpolated_grid_intensity, common_y
    

# example usage:
#
#filepath = # enter file path to main folder
#filepath += '/monitors/'
#
#k_point_folders = ['0/','0_0500/', '0_1000/','0_1500/', '0_2000/', '0_2500/', '0_3000/', '0_3500/', '0_4000/', '0_4500/', '0_5000/', '0_5500/', '0_6000/', '0_6500/', '0_7000/', '0_7500/', '0_8000/', '0_8500/', '0_9000/', '0_9500/', '1/']
#k_point_values = np.linspace(0,1,21)
#
#image, y_coords = band_structure_FDTD(filepath, k_point_folders, k_point_values, ymin=0, ymax=8, imshowplot=True, logscale=True, vmin=2, vmax=8267, xlabel=['$\Gamma$','$\mathrm{L}$'])
