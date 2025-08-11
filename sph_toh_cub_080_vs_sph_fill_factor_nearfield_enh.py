#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:13:22 2025

@author: winniep01
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.constants import hbar, e
import h5py

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

#%% load 080 fill factor

filepath_dir = "/home/winniep01/Desktop/sheldon-scratch/big_nearfield_data_files/"

E_total_cub_080 = np.load(filepath_dir + "nearfield_data_cub_080_stair_4D_fcc_abs.npy")

freq_cubes_080 = np.load(filepath_dir + "cub_sc_f_axis.npy")

E_total_toh_080 = np.load(filepath_dir + "nearfield_data_toh_080_stair_4D_fcc_abs.npy")

freq_toh_080 = np.load(filepath_dir + "toh_sc_f_axis_080.npy")

#%%

x_cubes_080 = np.load(filepath_dir + "cub_x_axis_080.npy")[0]
y_cubes_080 = np.load(filepath_dir + "cub_y_axis_080.npy")[0]
z_cubes_080 = np.load(filepath_dir + "cub_z_axis_080.npy")[0]

#%% import spheres

E_total_sph = np.load(filepath_dir + "nearfield_data_sph_4D_fcc_abs.npy")
#%%
freq_sph = np.load(filepath_dir + "sph_f_axis.npy")

#%% sum E fields

sum_E_cub_080 = np.sum(E_total_cub_080, axis=(1,2,3)) / np.size(E_total_cub_080)
sum_E2_cub_080 = np.sum(E_total_cub_080**2, axis=(1,2,3)) / np.size(E_total_cub_080)
sum_E4_cub_080 = np.sum(E_total_cub_080**4, axis=(1,2,3)) / np.size(E_total_cub_080)

sum_E_toh_080 = np.sum(E_total_toh_080, axis=(1,2,3)) / np.size(E_total_toh_080)
sum_E2_toh_080 = np.sum(E_total_toh_080**2, axis=(1,2,3)) / np.size(E_total_toh_080)
sum_E4_toh_080 = np.sum(E_total_toh_080**4, axis=(1,2,3)) / np.size(E_total_toh_080)
#%%
sum_E_sph = np.sum(E_total_sph, axis=(1,2,3)) / np.size(E_total_sph)
sum_E2_sph = np.sum(E_total_sph**2, axis=(1,2,3)) / np.size(E_total_sph)
sum_E4_sph = np.sum(E_total_sph**4, axis=(1,2,3)) / np.size(E_total_sph)

#%% save data

comb_array = ['sum_E_cub_080', 'sum_E2_cub_080', 'sum_E4_cub_080', 'sum_E_toh_080', 'sum_E2_toh_080', 'sum_E4_toh_080', 'sum_E_sph', 'sum_E2_sph', 'sum_E4_sph']

for i in comb_array:
    filep = f"/home/winniep01/Storage/thesis/lumerical/nearfield_analysis_space_filling/{i}.npy"
    np.save(filep, eval(i))

#%% load data

comb_array = ['sum_E_cub_080', 'sum_E2_cub_080', 'sum_E4_cub_080', 'sum_E_toh_080', 'sum_E2_toh_080', 'sum_E4_toh_080', 'sum_E_sph', 'sum_E2_sph', 'sum_E4_sph']

for i in comb_array:
    filep = f"/home/winniep01/Storage/thesis/lumerical/nearfield_analysis_space_filling/{i}.npy"
    exec(f"{i} = np.load(filep)")


#%% plot E field sums


fig, ax = plt.subplots(ncols=2)
plt.tick_params(direction="in")
ax[0].tick_params(direction="in")


ax[0].plot(freq_cubes_080[0]*(hbar/e*(2*np.pi)), sum_E2_cub_080)
ax[0].plot(freq_toh_080[0]*(hbar/e*(2*np.pi)), sum_E2_toh_080)
ax[0].plot(freq_sph[0]*(hbar/e*(2*np.pi)), sum_E2_sph)

ax[1].plot(freq_cubes_080[0]*(hbar/e*(2*np.pi)), sum_E4_cub_080)
ax[1].plot(freq_toh_080[0]*(hbar/e*(2*np.pi)), sum_E4_toh_080)
ax[1].plot(freq_sph[0]*(hbar/e*(2*np.pi)), sum_E4_sph)

ax[0].set_xlabel("energy (eV)")
ax[1].set_xlabel("energy (eV)")
ax[0].set_ylabel("$\sum_{x,y,z} |E(x,y,z,\omega)|^2$")
ax[1].set_ylabel("$\sum_{x,y,z} |E(x,y,z,\omega)|^4$")

plt.show()


