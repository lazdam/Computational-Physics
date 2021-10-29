'''
Problem Set #5, Problem #2
Mattias Lazda
260845451
Last Updated: October 23, 2021
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-muted')

def get_correlation_func(f, g):
    '''
    Function to compute the correlation function of f and g.
    '''

    F = np.fft.rfft(f)
    G = np.fft.rfft(g)

    corr = np.fft.irfft(F*np.conj(G))

    return corr



x = np.linspace(-5,5,1000)
y = np.exp(-0.5*(x)**2/0.35**2)
corr_func = get_correlation_func(y, y)

# Plot
if True: 
    plt.plot(y, label = 'Original')
    plt.plot(np.abs(corr_func), label = 'Correlation')
    plt.legend()
    plt.savefig('figures/correlation_gaussian.png')
    plt.show()




    