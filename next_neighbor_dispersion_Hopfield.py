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
rabi = 0.1*(e/hbar)

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
    #c = 0
    #b = 0
    #b1 = 0
    #a = 0
    #a1 = 0
    E1 = hbar*wpt
    E11 = hbar*wptuk
    E2 = hbar*wmk
    
    Hmatrix = [[2*b+E1,2*c,a,2*b,2*c,-a], [2*c,2*b1+E11,a1,2*c,2*b1,-a1], [-a,-a1,E2,-a,-a1,0], [-2*b,-2*c,-a,-2*b-E1,-2*c,a], [-2*c,-2*b1,-a1,-2*c,-2*b1-E11,a1], [-a,-a1,0,-a,-a1,-E2]]

    return Hmatrix

k_values = np.linspace(np.sqrt(3)*np.pi/a*0.00001, np.sqrt(3)*np.pi/a, 500)  # Example: 500 points between -10 and 10

# Store the eigenvalues
eigenvalues = []

# Compute the eigenvalues for each t
color_array = np.zeros((len(k_values), 6, 3))
# Compute the eigenvalues for each t
for idx, k in enumerate(k_values):
    eigvals, eigvects = np.linalg.eig(matrix(k, rabi, wmk, a))
    eigenvalues.append(eigvals)
    
    c_pt = np.sqrt(eigvects[0]**2 + eigvects[1]**2 + eigvects[3]**2 + eigvects[4]**2)
    c_pl = np.sqrt(eigvects[2]**2 + eigvects[5]**2)
    
    color_array[idx,:,2] = c_pl
    color_array[idx,:,1] = c_pt

# Convert the list of eigenvalues into a NumPy array for easier manipulation
eigenvalues = np.array(eigenvalues)
fig, ax = plt.subplots(ncols=2, figsize=(3,4))


# Plot the eigenvalues
for i in range(eigenvalues.shape[1]):  # Loop over each eigenvalue (column of the matrix)
    ax[0].scatter(k_values/(np.sqrt(3)*np.pi/a), eigenvalues[:, i]/e, label=f'Eigenvalue {i+1}', s=0.1, color=color_array[:,i,:], rasterized=True)

# Store the eigenvalues
eigenvalues = []

wmk = 2*(e/hbar)
rabi = 1*(e/hbar)

color_array = np.zeros((len(k_values), 6, 3))
# Compute the eigenvalues for each value of t
for idx, k in enumerate(k_values):
    eigvals, eigvects = np.linalg.eig(matrix(k, rabi, wmk, a))
    eigenvalues.append(eigvals)
    #eigvects = eigvects.T
    
    c_pt = np.sqrt(np.abs(eigvects[0])**2 + np.abs(eigvects[1])**2 + np.abs(eigvects[3])**2 + np.abs(eigvects[4])**2)
    c_pl = np.sqrt(eigvects[2]**2 + eigvects[5]**2)
    
    color_array[idx,:,2] = c_pl
    color_array[idx,:,1] = c_pt

color_array /= np.max(color_array)

# Convert the list of eigenvalues into a NumPy array for easier manipulation
eigenvalues = np.array(eigenvalues) 


# Plot the eigenvalues
for i in range(eigenvalues.shape[1]):  # Loop over each eigenvalue (column of the matrix)
    ax[1].scatter(k_values/(np.sqrt(3)*np.pi/a), eigenvalues[:, i]/e, label=f'Eigenvalue {i+1}', s=1, c=color_array[:,i,:], rasterized=True)



nticks = 1
ticks = []
ticks_label = ["$\Gamma$", r"L"]

for i in range(nticks + 1):
    ticks.append((i/nticks))


# put in photon lines
xph = np.linspace(0, np.sqrt(3)*np.pi/a, 20)
yph = np.abs(xph)*cl/n_eff

ax[1].plot(xph/(np.sqrt(3)*np.pi/a), yph*(hbar/e), '--', color='green')

for i in [0, 1]:
    
    ax[i].plot(xph/(np.sqrt(3)*np.pi/a), yph*(hbar/e), '--', color='green')
    yph = cl * (2*np.sqrt(3)*np.pi/a - xph)/n_eff
    ax[i].plot(xph/(np.sqrt(3)*np.pi/a), yph*(hbar/e), '--', color='green')    
    # put in w_mk
    y_wmk = [wmk*(hbar/e), wmk*(hbar/e)]
    x_wmk = [-2*np.pi/a, 2*np.pi/a]
    ax[i].plot(x_wmk, y_wmk, '--', color='blue')
        
    ax[i].set_xlabel('')
    ax[0].set_ylabel('Energy (eV)')
    ax[i].set_ylim(0, 4.25)
    ax[i].set_xlim(0, 1)
    nticks = 1
    ticks = []
    ticks_label = ["$\Gamma$", "$\mathrm{L}$"]
    
    for i in range(nticks + 1):
        ticks.append((i/nticks))
            
    ax[i].set_xticks(ticks, ticks_label)
    ax[i].tick_params(direction='in')

ax[0].set_title("$\eta = 0.05$")
ax[1].set_title("$\eta = 0.5$")

ticks = []
ticks_label = ["$\Gamma$", "$\mathrm{L}$"]


for i in range(nticks + 1):
    ticks.append((i/nticks))


ax[0].text(0.05, 2.15, "$\omega_\mathrm{pl}$", color="blue")
ax[1].text(0.05, 2.15, "$\omega_\mathrm{pl}$", color="blue")
ax[0].text(0.05, .5, "$\omega_\mathrm{pt}$", color="green")
ax[1].text(0.05, .5, "$\omega_\mathrm{pt}$", color="green")
ax[1].text(0.1, 3, "$\omega_\mathrm{pt}^\mathrm{uk}$", color="green")
ax[0].text(0.1, 3, "$\omega_\mathrm{pt}^\mathrm{uk}$", color="green")

ax[1].text(0.7, 0.25, "$\omega_\mathrm{LP}$", color="red")
ax[1].text(0.35, 1.4, "$\omega_\mathrm{IP}$", color="red")
ax[1].text(0.65, 3.9, "$\omega_\mathrm{UP}$", color="red")

ax[0].text(0.7, 0.8, "$\omega_\mathrm{LP}$", color="red")
ax[0].text(0.45, 1.6, "$\omega_\mathrm{IP}$", color="red")
ax[0].text(0.65, 2.45, "$\omega_\mathrm{UP}$", color="red")


ax[0].set_xticks(ticks, ticks_label)
ax[0].tick_params(direction='in')

ax[-1].yaxis.tick_right()
ax[0].set_yticks(np.linspace(0, 5, num=6))
ax[1].set_yticks(np.linspace(0, 5, num=6))

q_fdtd = np.array([1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20])/20
E_fdtd = [0.05508767082175103, 0.12304430716880366, 0.19442656055600832, 1.89124017995389915, 0.25575951275904415, 1.87556976023299, 0.322067534585462946, 1.8660971856863589, 0.38835826017107844, 1.8447017358126228, 0.4506162580939844, 1.8224924216245593, 0.5171917306820594, 1.7949595358965589, 0.5755809729674707, 1.7678558455429458, 0.6310380983969843, 1.7379409923064488, 0.693107419550786, 1.7172512185885147, 0.7448318538456209, 1.686216558011614, 0.7914238378302723, 1.6547952972814783, 0.8476316834390172, 1.6339887873523222, 0.8822682464576307, 1.602280263681674, 0.9270796142517297, 1.5820815156252344, 0.9573104712381992, 1.5619276109675881, 0.9774643758958454, 1.551850658638765, 0.9938194040557219, 1.5414341777190788, 0.9938194040557219, 1.5312931633919795]

ax[0].set_ylim(0, 4.25)
ax[1].set_ylim(0, 4.25)

#ax[0].scatter(q_fdtd, E_fdtd, color="purple", s=5)

#plt.grid(True)
plt.savefig("graph.pdf",
            dpi=1000,
            bbox_inches='tight',
            )

#plt.tight_layout()
plt.show()