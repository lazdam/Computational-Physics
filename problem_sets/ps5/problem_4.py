'''
Problem Set #5, Problem #4
Mattias Lazda
260845451
Last Updated: October 29, 2021
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-muted')


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
        # Pad both arrays with zeros
        diff = npts_f - npts_g
        f_pad = pad_fun(f, npts_f)
        g_pad = pad_fun(g, npts_f + diff)

    else:         
        # Pad both arrays with zeros
        diff = npts_g - npts_f
        f_pad = pad_fun(f, npts_g + diff)
        g_pad = pad_fun(g, npts_g)


    # Get convolution
    f_fft = np.fft.fft(f_pad)
    g_fft = np.fft.fft(g_pad)

    convolution = np.fft.ifft(f_fft*g_fft)

    return np.abs(convolution)


# Test function
x = np.linspace(-5,5,1000)
y = np.exp(-0.5*(x)**2/0.35**2)

shifts = np.linspace(0, 1.9999, 100)

for shift in shifts: 
    plt.clf()
    g = np.zeros(2*len(y))
    g[int(shift*len(y))] = 1
    conv = conv_safe(y, g)
    plt.plot(conv)
    plt.title('Shifted by {0} frames'.format(int(shift*len(y))))
    plt.pause(0.01)

print('Length of Output:{0}'.format(len(conv)))

