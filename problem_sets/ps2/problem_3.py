import numpy as np 
from numpy.polynomial.chebyshev import chebfit, chebval

def model_log2(npts=20, ord=19, tol=1e-6):
    '''
    Models the log2 function in x e [0.5, 1]

    Parameters: 
    -----------
    npts: int
        Number of points to fit through. Default 20. 
    ord: int
        Maximum order of Chebyshev Polynomial used. Default set to 19.
    tol: float
        Error tolerance on Chebyshev polynomial fit. Default 1e-6 
    
    Returns: np.array
        Array of truncated Chebyshev polynomial coefficients. 
    '''
    
    #Generate points to fit 
    x = np.linspace(0.5, 1, npts) 
    y = np.log2(x)

    #Re-scale x to be between (-1,1)
    x_scaled = 4*x - 3

    #Generate coeffs
    coeffs = chebfit(x_scaled, y, ord)
    
    #Truncate coefficients. If coeff is less than error tolerance, remove it. 
    coeffs[np.abs(coeffs)<=tol] = 0 
    
    #Calculate number of removed coefficients. 
    num_removed_coeffs = len(np.where(coeffs == 0)[0])
    print('Began with {0} coefficients but removed {1} of them. Error tolerance satisfied.'.format(len(coeffs),num_removed_coeffs))
    
    return coeffs


def mylog2(x):

    '''
    Computes the natural logarith of any number with accuracy specified in model_log2.

    Parameters: 
    -----------
    x: int
        x value you'd like to evaluate the natural log of. 

    Returns: int
        ln(x)
    '''
    
    M, exp = np.frexp(x) #M between (0.5, 1), exp an integer
    
    #Get coefficients using default fit
    coeffs = model_log2() 
    
    #Re-scale M to be between -1 and 1
    M_scaled = 4*M - 3 

    #Calculate predicted value using Chebyshev Coefficients
    log2_pred = chebval(M_scaled, coeffs)
    
    #Calculate natural log using equation (6) in attached PDF
    log_e = (log2_pred + exp)/1.4426950408889634
    
    return log_e


x = 0.0001
y = mylog2(x)
y_true = np.log(x)

print('My natural log is: {0} with true error {1}.'.format(y, np.abs(y - y_true)))
    