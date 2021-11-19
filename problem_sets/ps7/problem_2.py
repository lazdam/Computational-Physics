"""
Problem Set 7, Problem 2
Mattias Lazda
260845451
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
plt.style.use('seaborn-muted')

# Define all the different PDF's that we'll use for this problem 

def lorentz_PDF(x, gamma = 1):
    return 1/(gamma*np.pi) * 1/(1 + (x/gamma)**2)

def gauss_PDF(x, sigma = 1):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5 *(x/sigma)**2)

def pow_PDF(x, alpha = 1.5):
    return x**(-alpha)

def exp_PDF(x, lamda = 1):
    return lamda*np.exp(-lamda*x)


# First, let's determine which of the three distributions are viable to use as a bounding function.

# Scaling factors, adjust if needed 
M_g = 15
M_l = np.pi
M_p = 1

# Plot to see how the distributions look, allows us to choose which bounding fnc to use
if True: 
    alpha = 1.5
    gamma = 1
    lamda = 1
    sigma = 5

    plt.figure()
    xs = np.linspace(0,100,10000)
    bool_set = set(exp_PDF(xs) <= M_l*lorentz_PDF(xs, gamma = gamma))
    print(bool_set)
    if len(bool_set)!=1:
        print('Check gamma!')
        assert(1==0)

    plt.plot(xs,exp_PDF(xs, lamda = lamda ),label='Exponential', ls = '--')
    plt.plot(xs,M_g*gauss_PDF(xs, sigma = sigma), label='Gaussian')
    plt.plot(xs,M_l*lorentz_PDF(xs, gamma = gamma), label='Lorentzian')
    plt.plot(xs[80:],M_p*pow_PDF(xs, alpha = alpha)[80:], label='Power Law') # Neglect power law
    plt.legend()
    plt.xlim(0,30)
    plt.title('Comparing Candidate Bounding Distributions')
    plt.savefig('Figures/comparison_of_distributions.png')
    plt.show()
    plt.clf()

# From the generated plot and from general theory, we conclude that the only the Lorentzian can be used
# as a bounding function. Therefore, for the rejection method, we will optimize the Lorentzian dist
# to generate exponential deviates.

# Generate lorentzian distributed values
# This comes from inverting the CDF of the PDF above
def rand_lor(n, gamma = 1):
    q = np.random.rand(n)
    return gamma*np.tan(np.pi*(q - 0.5))

n = 10000000


# Generate random Lorentzian numbers
xs = rand_lor(n, gamma = gamma)
# Limit values to be greater than 0
xs = xs[xs>=0]

# Generate uniform numbers
us = np.random.rand(len(xs))

# Get exponential and lorentzian probabilities
exps = exp_PDF(xs)
lors = lorentz_PDF(xs, gamma = gamma)

# Rejection step
keep = us < exps / (M_l * lors)
exp_rands = xs[keep]

# Plot histogram 
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
    plt.savefig('Figures/exp_from_lorentz.png')
    plt.show()
    print(f"Method is {np.mean(keep)*100}% Efficient")


    

