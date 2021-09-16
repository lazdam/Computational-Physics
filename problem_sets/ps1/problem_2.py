import numpy as np
import matplotlib.pyplot as plt

def ndiff(fun, x, full = False):

    '''
    Computes numerical derivative of function.

    Parameters:
    -----------
    fun: Function (callable)
        Function you'd like to evaluate the derivative of. 
    x: ndarray or float
        Array of x values to evaluate numerical derivative at. Supports both arrays and single float values
    full: Bool
        Default set to False. If True, function returns derivative, dx and an estimate of error on the derivative. Else, returns derivative only. 

    '''
    
    #Prepare x values. If not already in array, put into one. 
    if type(x) == type(np.array([])):
        pass
    else: 
        x = np.array([x])

    eps = 2**-52 #default for 64-bit machine

    npt = len(x)
 
    #Determine step size, h. Requires estimate of third derivative. 

    h_approx = 0.0001 #Rough estimate to compute best estimate.     

    #Calculate approx third derivative for each x
    approx_3d = calculate_third_derivative(fun, x, h_approx)
    f0 = fun(x)

    best_deriv = np.zeros(npt)
    err_deriv = np.zeros(npt)
    best_h = np.zeros(npt)

    for i, f3_i in enumerate(approx_3d):

        if np.abs(f3_i) < eps: #This checks if third derivative is zero. If yes, use approximate solutions since h would blow up to infinity otherwise.

            h = eps**(1/3)
            num_deriv = (fun(x[i] + h) - fun(x[i] - h))/(2*h)
            error = eps**(2/3)

            best_deriv[i], err_deriv[i], best_h[i] = num_deriv, error, h


        else: #If third derivative isn't too small for the computer, compute h, numerical derivative and error according to equations (5.7.7/8/9) in Numerical Recipes. 
            
            
            h = np.cbrt(eps*f0[i]/f3_i)
            num_deriv = (fun(x[i] + h) - fun(x[i] - h))/(2*h)
            
            if np.abs(num_deriv) < eps: #I.e. when derivative is zero, can't avoid DivideByZero error when calculating error. Take rough estimate instead. 
                error = eps**(2/3)

            else: 
                error = np.cbrt((eps**2)*(f0[i]**2)*f3_i)/(num_deriv)

            best_deriv[i], err_deriv[i], best_h[i] = num_deriv, error, h


    if full: 

        return best_deriv, err_deriv, best_h, 

    else: 

        return best_deriv


def calculate_third_derivative(fun, x, h):
    '''
    Computes approximate third derivative using centered difference approximation (see equation (12) in attached PDF). 

    Parameters: 
    -----------
    fun: function
        Input function
    x: ndarray/float
        x values to evaluate derivative at
    h: step size

    Returns: ndarray/float
    '''

    f1 = fun(x + 2*h)
    f2 = -2*fun(x + h)
    f3 = 2*fun(x - h)
    f4 = -fun(x - 2*h)

    return (f1 + f2 + f3 + f4)/(2*h**3)


#Define a function to test
def fun(x):
    return np.cos(x)



x = np.pi/3
deriv, error, h  = ndiff(fun, x , full = True)
print('True Fractional Error:',deriv/-np.sin(x) - 1)
print('Estimated Fractional Error: ', error)





