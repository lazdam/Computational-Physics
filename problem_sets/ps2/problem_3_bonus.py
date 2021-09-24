import numpy as np 
from numpy.polynomial.legendre import legfit, legval
from numpy.polynomial.chebyshev import chebfit, chebval


def model_log2_chebyshev_legendre(npts=20, ord=19, tol=1e-6):
    '''
    Models the log2 function in x e [0.5, 1]

    Parameters: 
    -----------
    npts: int
        Number of points to fit through. Default 20. 
    ord: int
        Maximum order of Chebyshev and Legendre Polynomial used. Default set to 19.
    tol: float
        Error tolerance on Chebyshev and Legendre polynomial fit. Default 1e-6 
    
    Returns: np.array
        Array of truncated Chebyshev and Legendre polynomial coefficients. 
    '''
    
    #Generate points to fit 
    x = np.linspace(0.5, 1, npts) 
    y = np.log2(x)

    #Re-scale x to be between (-1,1)
    x_scaled = 4*x - 3

    #Generate coeffs
    coeffs_cheb = chebfit(x_scaled, y, ord)
    err_cheb = np.copy(coeffs_cheb)

    coeffs_legendre = legfit(x_scaled, y, ord)
    err_legendre = np.copy(coeffs_legendre) 

    #Truncate coefficients. If coeff is less than error tolerance, remove it. 
    coeffs_cheb[np.abs(coeffs_cheb)<=tol] = 0 
    coeffs_legendre[np.abs(coeffs_legendre)<=tol] = 0

    #Calculate max error by summing over truncated coefficients
    err_cheb[err_cheb>tol]=0
    max_err_cheb = np.sum(err_cheb)

    err_legendre[err_legendre>tol]=0
    max_err_legendre = np.sum(err_legendre)



    return coeffs_cheb, coeffs_legendre, max_err_cheb, max_err_legendre


def mylog2(x, npts = 20, ord = 19, tol = 1e-6):

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
    coeffs_cheb, coeffs_legendre, max_err_cheb, max_err_legendre = model_log2_chebyshev_legendre(npts = 20, ord = 19, tol = 1e-6) 
    
    #Re-scale M to be between -1 and 1
    M_scaled = 4*M - 3 

    #Calculate predicted value using Chebyshev/Legendre Coefficients
    log2_pred_cheb = chebval(M_scaled, coeffs_cheb)
    log2_pred_legendre = legval(M_scaled, coeffs_legendre)
    
    #Calculate natural log using equation (6) in attached PDF
    log_e_cheb = (log2_pred_cheb + exp)/1.4426950408889634
    log_e_legendre = (log2_pred_legendre + exp)/1.4426950408889634
    

    return log_e_cheb, log_e_legendre, max_err_cheb, max_err_legendre


x = 3
npts, ord, tol = 20, 19, 1e-6
y_cheb, y_leg, err_cheb, err_legendre = mylog2(x, npts, ord, tol)
y_true = np.log(x)


print('''For x = {0} and using both Chebyshev and Legendre Polynomial fitting through {1} points
and using a max order of {2}, we get: 

[Chebyshev]: ln(x) = {3} with max error {4}. RMS error = {5}.
[Legendre]: ln(x) = {6} with max error {7}. RMS error = {8}. 

'''.format(x, npts, ord, y_cheb, np.abs(err_cheb), np.abs(y_true - y_cheb), y_leg, np.abs(err_legendre), np.abs(y_true - y_leg) ))