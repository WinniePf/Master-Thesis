#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:58:30 2024

@author: winnie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, c, e


cl = c
d = 500e-9
a = d * np.sqrt(2)
wmk = 2*(e/hbar)
rabi = 1*(e/hbar)

F = 0.74*(280/330)**3
n_eff = (1-F) + F

def matrix(k, rabi, wmk, a):
    wpt = np.abs(k)*cl/n_eff
    wptuk = cl * (2*np.sqrt(3)*np.pi/a - k)/n_eff 
    a = 1j*hbar*rabi*np.sqrt(wmk/wpt)
    a1 = 1j*hbar*rabi*np.sqrt(wmk/wptuk)
    b = (hbar*rabi**2)/wpt
    b1 = (hbar*rabi**2)/wptuk
    c = (hbar*rabi**2)/np.sqrt(wptuk*wpt)
    E1 = hbar*wpt
    E11 = hbar*wptuk
    E2 = hbar*wmk
    
    Hmatrix = [[2*b+E1,2*c,a,2*b,2*c,-a], [2*c,2*b1+E11,a1,2*c,2*b1,-a1], [-a,-a1,E2,-a,-a1,0], [-2*b,-2*c,-a,-2*b-E1,-2*c,a], [-2*c,-2*b1,-a1,-2*c,-2*b1-E11,a1], [-a,-a1,0,-a,-a1,-E2]]

    return Hmatrix


def calc_evals(k_values, rabi, wmk, a):
    # Store the eigenvalues
    eigenvalues = []
    color_array = np.zeros((len(k_values), 6, 3))

    # Compute the eigenvalues for each t
    for idx, k in enumerate(k_values):
        eigvals, eigvects = np.linalg.eig(matrix(k, rabi, wmk, a))
        eigenvalues.append(eigvals)
        
        c_pt = np.sqrt(np.abs(eigvects[0])**2 + np.abs(eigvects[1])**2 + np.abs(eigvects[3])**2 + np.abs(eigvects[4])**2)
        c_pl = np.sqrt(np.abs(eigvects[2])**2 + np.abs(eigvects[5])**2)
        
        color_array[idx,:,2] = c_pl
        color_array[idx,:,1] = c_pt

    eigenvalues = np.array(eigenvalues)
    return eigenvalues, color_array


def plot_evals(eigenvalues, eigenvalues2, k_values, color_array, color_array2):
    fig, ax = plt.subplots(ncols=2, figsize=(3,4))

    # Plot the eigenvalues
    for i in range(eigenvalues.shape[1]):  # Loop over each eigenvalue (column of the matrix)
        ax[0].scatter(k_values/(np.sqrt(3)*np.pi/a), eigenvalues[:, i]/e, label=f'Eigenvalue {i+1}', s=0.1, c=color_array[:,i,:]/np.max(color_array), rasterized=True)
        ax[1].scatter(k_values/(np.sqrt(3)*np.pi/a), eigenvalues2[:, i]/e, label=f'Eigenvalue {i+1}', s=0.1, c=color_array2[:,i,:]/np.max(color_array2), rasterized=True)

    nticks = 1
    ticks = []
    ticks_label = [r"$\Gamma$", r"\mathrm{L}"]

    for i in range(nticks + 1):
        ticks.append((i/nticks))

    # put in photon lines
    xph = np.linspace(0, np.sqrt(3)*np.pi/a, 20)
    yph = np.abs(xph)*cl/n_eff
    yphuk = cl * (2*np.sqrt(3)*np.pi/a - xph)/n_eff
    # put in w_mk
    y_wmk = [wmk*(hbar/e), wmk*(hbar/e)]
    x_wmk = [-2*np.pi/a, 2*np.pi/a]

    nticks = 1
    ticks = []
    ticks_label = [r"$\Gamma$", r"$\mathrm{L}$"]
        
    for i in range(nticks + 1):
        ticks.append((i/nticks))

    for i in [0, 1]:
        ax[i].plot(xph/(np.sqrt(3)*np.pi/a), yph*(hbar/e), '--', color='green')
        ax[i].plot(xph/(np.sqrt(3)*np.pi/a), yphuk*(hbar/e), '--', color='green')    
        ax[i].plot(x_wmk, y_wmk, '--', color='blue')
            
        ax[i].set_xlabel('')
        ax[0].set_ylabel('Energy (eV)')
        ax[i].set_ylim(0, 4.25)
        ax[i].set_xlim(0, 1)
                
        ax[i].set_xticks(ticks, ticks_label)
        ax[i].tick_params(direction='in')

    ax[0].set_title(r"$\eta = 0.05$")
    ax[1].set_title(r"$\eta = 0.5$")

    ax[0].text(0.05, 2.15, r"$\omega_\mathrm{pl}$", color="blue")
    ax[1].text(0.05, 2.15, r"$\omega_\mathrm{pl}$", color="blue")
    ax[0].text(0.05, .5, r"$\omega_\mathrm{pt}$", color="green")
    ax[1].text(0.05, .5, r"$\omega_\mathrm{pt}$", color="green")
    ax[1].text(0.1, 3, r"$\omega_\mathrm{pt}^\mathrm{uk}$", color="green")
    ax[0].text(0.1, 3, r"$\omega_\mathrm{pt}^\mathrm{uk}$", color="green")

    ax[1].text(0.7, 0.25, r"$\omega_\mathrm{LP}$", color="red")
    ax[1].text(0.35, 1.4, r"$\omega_\mathrm{IP}$", color="red")
    ax[1].text(0.65, 3.9, r"$\omega_\mathrm{UP}$", color="red")

    ax[0].text(0.7, 0.8, r"$\omega_\mathrm{LP}$", color="red")
    ax[0].text(0.45, 1.6, r"$\omega_\mathrm{IP}$", color="red")
    ax[0].text(0.65, 2.45, r"$\omega_\mathrm{UP}$", color="red")


    ax[0].set_xticks(ticks, ticks_label)
    ax[0].tick_params(direction='in')

    ax[-1].yaxis.tick_right()
    ax[0].set_yticks(np.linspace(0, 5, num=6))
    ax[1].set_yticks(np.linspace(0, 5, num=6))

    ax[0].set_ylim(0, 4.25)
    ax[1].set_ylim(0, 4.25)

    plt.savefig("next_BZ_dispersion.pdf",
                dpi=1000,
                bbox_inches='tight',
                )

    plt.show()


k_values = np.linspace(np.sqrt(3)*np.pi/a*0.00001, np.sqrt(3)*np.pi/a, 500)  # Example: 500 points between -10 and 10

evals1, c_array1 = calc_evals(k_values, rabi/10, wmk, a)
evals2, c_array2 = calc_evals(k_values, rabi, wmk, a)

plot_evals(evals1, evals2, k_values, c_array1, c_array2)

