"""
Problem Set 7, Problem 2
Mattias Lazda
260845451
"""

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-muted')


def exp_PDF(x, lamda = 1):
    return lamda*np.exp(-lamda*x)

N = 10000000
lamda = 1

# Calculated bounds, see attached PDF
umax = np.sqrt(lamda)
vmax = 2/np.e/np.sqrt(lamda)

# Generate random numbers
u = np.random.uniform(low=0, high = umax, size = N)
v = np.random.uniform(low = 0, high = vmax, size = N)

# Do rejection step
keep = u < np.sqrt(exp_PDF(v/u, lamda))
exp_rands = v[keep]/u[keep]

# Plot results
if True: 
    bins = np.linspace(min(exp_rands), max(exp_rands), 501)
    aa, bb = np.histogram(exp_rands, bins)
    aa = aa/aa.sum()
    cents = 0.5*(bins[1:]+bins[:-1])
    pred = exp_PDF(cents, lamda = 1)
    pred = pred/pred.sum()
    plt.plot(cents, aa, "*", label = 'Samples')
    plt.plot(cents, pred, 'r', label = 'Predicted Exponential')
    plt.title(f'Efficiency: {np.mean(keep)*100}%')
    plt.legend()
    plt.savefig('Figures/exp_from_lorentz_ratio_of_uni.png')
    plt.show()
    
