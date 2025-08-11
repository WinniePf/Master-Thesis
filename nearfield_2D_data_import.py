#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:03:08 2024

@author: winniep01
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "b", "c", "g", "m"]) 


def load_file(filepath):
    data = np.loadtxt(filepath, delimiter="\t", dtype='str', skiprows=0)
    xz = [[], []]
    intensities = []
    i = 0
    n = 3
    # load x, y, z range:
    for i in range(2):
        
        while True:
            try:
                xz[i].append(float(data[n])*10**9)
                n = n + 1
            except ValueError:
                n = n + 1
                break
    
    # load intensities:
    while True:
        try:
            intensities.append(np.array(data[n].split(), dtype=float))
            n = n + 1
        except IndexError:
            break
    
    return xz, np.array(intensities)

def openimage(filepath, max_intensity=0, title=None, cmap='RdBu', aspect='equal', y_offset=0, magnitude=False, figsize=None, n=1, m=1):
    a = load_file(filepath)
    norm_int = np.transpose(a[1])
    
    if figsize:
        plt.figure(figsize=figsize)
    else:    
        plt.figure(figsize=(2.8,5))
        
    plt.xlabel("x (nm)")
    plt.ylabel("z (nm)")
    
    norm_int = np.tile(norm_int, (n,m))
    
    
    if magnitude:
        plt.imshow(norm_int, cmap=cmap, extent=(np.min(a[0][0]), np.max(a[0][0]), np.min(a[0][1])+y_offset, np.max(a[0][1])+y_offset), interpolation='antialiased', aspect=aspect,origin='lower', vmin=0, vmax=max_intensity) # vmin=vminn, vmax=vmaxx, , vmin=-8.8, vmax=8.8) #
    else:
        plt.imshow(norm_int, cmap=cmap, extent=(np.min(a[0][0]), np.max(a[0][0]), np.min(a[0][1])+y_offset, np.max(a[0][1])+y_offset), interpolation='antialiased', aspect=aspect,origin='lower', vmin=-max_intensity, vmax=max_intensity) # vmin=vminn, vmax=vmaxx, , vmin=-8.8, vmax=8.8) #
 
    if title:
        plt.title(title)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$|E/E_0|$')
    plt.tight_layout()
    if filepath[-4] == ".":
        plt.savefig(f"{filepath[:-4]}.png", dpi=1000)
        plt.savefig(f"{filepath[:-4]}.pdf", dpi=1000)
    else:
        plt.savefig(f"{filepath}.png", dpi=1000)
        plt.savefig(f"{filepath}.pdf", dpi=1000)
    
    plt.show()


#%%

filename = 'nf_3_1424_e14.txt'

filepath = '/home/winniep01/Desktop/Lumerical Grouphome/hamburg_toh_structure/FCC_spheres_as_comparison/nearfield_cubes_24ML/' + filename

openimage(filepath, max_intensity=7, cmap='RdBu', aspect='auto')

