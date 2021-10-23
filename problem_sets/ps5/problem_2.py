'''
Problem Set #5, Problem #2
Mattias Lazda
260845451
Last Updated: October 23, 2021
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')

def get_correlation_func(f, g):
    '''
    Function to compute the correlation function given two functions.
    '''

    F = np.fft.rfft(f)
    G = np.fft.rfft(g)

    return np.fft.irfft(F*np.conj(G))


x = np.linspace(-5,5,1000)
y = np.exp(-0.5*(x)**2/0.35**2)
corr_func = get_correlation_func(y, y)

if True: 
    plt.plot(x, np.abs(corr_func))
    plt.savefig('figures/correlation_gaussian.png')
    plt.show()




    