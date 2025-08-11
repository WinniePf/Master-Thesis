import numpy as np

# load lumerical farfield absorption text file. Resulting numpy array has shape (3, 2, N), 
# where the first index corresponds to Transmission (1), Reflection (2), or Absorption (3). 
# Trans/Abs/Refl spectra then have shape (2, N) where the first index refers to the x or y axis 
# and each axis has N data points.

def load_file(filepath, delim=",", dtype='str', skiprows=3):
    data = np.loadtxt(filepath, delimiter=delim, dtype=dtype, skiprows=skiprows)
    TRA = [[[], []], [[], []], [[], []]]
    i = 0
    n = 0
    while True:
        try:
            TRA[n][1].append(float(data[i][1]))
            TRA[n][0].append(float(data[i][0]))
            i = i + 1
        except IndexError:
            break
        except ValueError:
            n = n + 1
            i = i + 1
        except FileNotFoundError:
            print("File not found")
            break
    return np.array(TRA)


# example plot to compare two TRA spectra. File paths of files need to be entered
'''
filepath1 = 
filepath2 = 

toh_sc = load_file(filepath1)
toh = load_file(filepath2)

import matplotlib.pyplot as plt

plt.figure(figsize=(3,4))

plt.tick_params("both", direction="in")
for i in [0,1,2]:
    plt.plot(toh_sc[i][0], np.array(toh_sc[i][1]) + i, linewidth=3, color="g")
    plt.plot(toh[i][0], np.array(toh[i][1]) + i, linewidth=3, color="b")

plt.xlim([0.1, 2.5])

plt.show()
'''
