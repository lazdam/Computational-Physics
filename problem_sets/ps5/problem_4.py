'''
Problem Set #5, Problem #4
Mattias Lazda
260845451
Last Updated: October 26, 2021
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')


def pad_fun(arr, npad):
    return np.hstack((arr, np.zeros(npad)))

def conv_safe(f, g):

    npts_f = len(f)
    npts_g = len(g)
    
    if npts_f == npts_g:
        # Pad end of both arrays with zeros. 
        f_pad = pad_fun(f, npts_f)
        g_pad = pad_fun(g, npts_g)
    
    elif npts_f > npts_g:
        # Truncate f to match same size as g
        f = f[:npts_g]
        
        # Pad both arrays with zeros
        f_pad = pad_fun(f, npts_g)
        g_pad = pad_fun(g, npts_g)

    else: 
        # Truncate g to match same size as f
        g = g[:npts_f]
        
        # Pad both arrays with zeros
        f_pad = pad_fun(f, npts_f)
        g_pad = pad_fun(g, npts_f)


    # Get convolution
    f_fft = np.fft.rfft(f_pad)
    g_fft = np.fft.rfft(g_pad)

    convolution = np.fft.irfft(f_fft*g_fft)

    return convolution


# Test function
x = np.linspace(-5,5,1000)
y = np.exp(-0.5*(x)**2/0.35**2)

shifts = np.linspace(0, 0.95, 4)

for shift in shifts: 
    g = np.zeros(len(y))
    g[int(shift*len(y))] = 1
    conv = conv_safe(y, g)
    plt.plot(conv, label = 'Shifted by {0} frames'.format(int(shift*len(y))))

plt.legend(loc = 'lower right')
plt.savefig('figures/conv_safe_shifted_example.png')
plt.show()


