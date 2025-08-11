#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 09:12:22 2025

@author: winniep01
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.constants import hbar,c,e
from scipy.optimize import curve_fit

N_sweep = 10
F = 0.6
diameters = np.linspace(50, 500, 10) * 1e-9
n_eff = 1.4 * (1 - F) + F
n_layers = 6

farfields = np.zeros((N_sweep,4,5000))


for i in range(N_sweep):
    filepath = f'Storage/thesis/lumerical/lattice_fcc_111/parameter_sweep_coupling_strength/6ML_cubes_fill_factor_sweep/farfield_{i+1}.txt'

    farfields[i] = np.loadtxt(filepath, dtype='float', delimiter='\t').T

peaks = []

for i in range(N_sweep):
    # determine fitting range
    E_max = 1.8*np.exp(-0.13*(i)) 

    max_idx = np.argmin(np.abs(E_max - farfields[0,0]))

    peak = find_peaks(farfields[i,3,:max_idx], prominence=0.01)[0]
    if i == 9:
        peak = find_peaks(farfields[i,3,:max_idx], height=0.3)[0]

    peaks.append(peak)


def hopfield_minus(k, w_R, w_pl):
    def w_pt(k):
        w = k * c / n_eff
        return w
    res = np.sqrt((w_pt(k)**2 + 4*w_R**2 + w_pl**2 - 1 * np.sqrt((w_pt(k)**2 + 4*w_R**2 + w_pl**2)**2 - 4*w_pt(k)**2*w_pl**2)) / 2)
    return res 

w_pl = np.zeros(N_sweep)
rabi = np.zeros(N_sweep)

for i in range(N_sweep):
    gap = diameters[i]/F**(1/3) - diameters[i]
    a = (diameters[i] + gap) * np.sqrt(2)
    k_max = np.sqrt(3)*np.pi/a
    peak = peaks[i]
    k_array = np.linspace(k_max/(n_layers - 1), len(peak)*k_max/(n_layers - 1), num=len(peak))
    popt = curve_fit(hopfield_minus, k_array, farfields[0,0,peak]*e/hbar, p0=[e/hbar,e/hbar])[0]
    w_pl[i] = popt[1]/e*hbar
    rabi[i] = popt[0]/e*hbar

    
    
print(np.round(rabi,2))
print(np.round(w_pl,2))

print(np.round(rabi/w_pl,2))

plt.figure()
plt.plot(diameters/1e-9, np.round(rabi/w_pl,2))
plt.xlabel('diameter')



