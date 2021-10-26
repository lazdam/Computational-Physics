'''
Problem Set #5, Problem #1
Mattias Lazda
260845451
Last Updated: October 23, 2021
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')

def shift_arr(arr, shift_frames):
    '''
    Shift an array set by the number of frames. 
    '''
    # Define functions to convolve
    f = arr
    
    # Define delta function
    g = np.zeros(len(f))
    g[shift_frames] = 1

    #Convolve functions
    F = np.fft.rfft(f)
    G = np.fft.rfft(g)
    shifted_func = np.fft.irfft(F*G)

    return shifted_func


x = np.linspace(-5,5,1000)
y = np.exp(-0.5*(x)**2/0.35**2)
num_frames = int(len(y)/2)
y_shifted = shift_arr(y, num_frames)

if True: 
    plt.plot(x, y, label = 'Original')
    plt.plot(x, np.abs(y_shifted), label = 'Shifted')
    plt.legend(loc='upper right')
    plt.savefig('figures/gauss_shift.png')
    plt.show()


