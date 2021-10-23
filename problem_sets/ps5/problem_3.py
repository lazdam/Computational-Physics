'''
Problem Set #5, Problem #3
Mattias Lazda
260845451
Last Updated: October 23, 2021
'''

import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('seaborn-pastel')

# Import function from Problem #1 
def shift_arr(arr, shift_frames):

    f = arr
    g = np.zeros(len(f))
    g[shift_frames] = 1
    F = np.fft.rfft(f)
    G = np.fft.rfft(g)
    shifted_func = np.fft.irfft(F*G)

    return shifted_func

# Import function from Problem #2
def get_correlation_func(f, g):

    F = np.fft.rfft(f)
    G = np.fft.rfft(g)

    return np.fft.irfft(F*np.conj(G))


# Combine functions
def modified_correlation(f, g, frame_shift):
    '''
    Computes the correlation function of f and g, where f is shifted by a set number of frames.
    '''
    shifted_f = shift_arr(f, frame_shift)

    return get_correlation_func(shifted_f, g), shifted_f

# Generate Gaussian
nframes = 1000
x = np.linspace(-5,5,nframes)
y = np.exp(-0.5*(x)**2/0.35**2)

# Test multiple frame shifts
frame_shift_arr = [int(nframes/4), int(nframes/2), int(3*nframes/4)]
for i in frame_shift_arr:
    plt.plot(x, modified_correlation(y, y, i)[0], label = 'Shifted Frames: {0}'.format(i))
    plt.legend(loc = 'lower center')

plt.savefig('correlated_shifted_gauss.png')    
plt.show();plt.clf()

for i in frame_shift_arr:
    plt.plot(x, modified_correlation(y, y, i)[1], label = 'Shifted Frames: {0}'.format(i))
    plt.legend(loc = 'lower center')
    
plt.savefig('shifted_gauss_p3.png')
plt.show()
