'''
Problem Set #5, Problem #3
Mattias Lazda
260845451
Last Updated: October 23, 2021
'''

import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('seaborn-muted')

# Import function from Problem #1 
def shift_arr(f, shift_frames):

    delta = np.zeros(len(f))
    delta[shift_frames] = 1
    F = np.fft.fft(f)
    G = np.fft.fft(delta)
    shifted_func = np.fft.ifft(F*G)

    return shifted_func

# Import function from Problem #2
def get_correlation_func(f, g):

    F = np.fft.fft(f)
    G = np.fft.fft(g)

    return np.fft.ifft(F*np.conj(G))


# Combine functions
def modified_correlation(f, g, frame_shift):

    shifted_f = shift_arr(f, frame_shift)
    #corr = get_correlation_func(g, shifted_f)
    corr = get_correlation_func(shifted_f, g)
    return np.abs(corr), np.abs(shifted_f)

# Generate Gaussian
nframes = 1000
x = np.linspace(-5,5,nframes)
y = np.exp(-0.5*(x)**2/0.35**2)

# Test multiple frame shifts
shift = np.linspace(0, 0.99, 100)

for i in shift:
    plt.clf()
    corr, shift = modified_correlation(y, y, int(i*nframes))
    plt.plot(corr, label = 'Correlation')
    plt.plot(y, label = 'Original')
    plt.plot(shift, label = 'Shifted', ls = '--')
    plt.title('Shifted by {0} Frames'.format(int(i*nframes)))
    plt.legend()
    plt.pause(0.001)
    




