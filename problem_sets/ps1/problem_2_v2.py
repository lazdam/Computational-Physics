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


    #STEP 1: 
    #Determine step size, h. Requires estimate of third derivative. 

    h_approx = eps**(1/3) #Rough estimate to compute best estimate. 

    #Define function evalulations to compute approximate third derivative.
    #See equation (13) in "Problem Set 1 - Mattias Lazda" (PDF)

    f1 = fun(x + 2*h_approx)
    f2 = -2*fun(x + h_approx)
    f3 = 2*fun(x - h_approx)
    f4 = -fun(x - 2*h_approx)

    #Calculate approx third derivative for each x
    third_deriv = (f1 + f2 + f3 + f4)/(2*h_approx**3)

    eps_arr = np.full((len(third_deriv)), h_approx)
    
    #Deal with the case when 3rd derivative is zero. In that case, to avoid DivideByZero errors,
    #if 3_deriv is smaller than eps, set 3_deriv to a very small number 
    third_deriv_use = np.max(np.vstack((third_deriv,eps)), axis = 0)


    #Compute best h value (see equation (5.7.8) in Numerical Recipes)
    h = np.cbrt(fun(x)*eps/third_deriv_use)

    #STEP 2: 
    #Calculate centered derivatives and respective errors. 

    deriv = (fun(x + h) - fun(x - h))/(2*h)

    print('Error: ', deriv/-np.sin(x) - 1)

    if full: 
        f = fun(x)

        #Calculate best_3_deriv given h
        f1 = fun(x + 2*h)
        f2 = -2*fun(x + h)
        f3 = 2*fun(x - h)
        f4 = -fun(x - 2*h)

        best_3_deriv = (f1 + f2 + f3 + f4)/(2*h**3)

        eps_arr = np.full((len(best_3_deriv)), h_approx)
        third_deriv_use = np.max(np.vstack((best_3_deriv,eps_arr)), axis = 0) #deal with best deriv = 0


        err_deriv = (eps**(2/3))*(f**(2/3))*(np.cbrt(third_deriv_use))/deriv #Fractional error according to equation (5.7.9) in Numerical Recipes.

        return deriv, h, err_deriv

    return deriv


#Define a function to test
def fun(x):
    return np.cos(x)


x = 5

deriv = ndiff(fun, x, full = True)
print(deriv)








