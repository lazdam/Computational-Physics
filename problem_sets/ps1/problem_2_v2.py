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


    #STEP 1: 
    #Determine step size, h. Requires estimate of third derivative. 

    h_approx = eps**(1/3) #Rough estimate to compute best estimate.     

    #Calculate approx third derivative for each x
    approx_3d = calculate_third_derivative(fun, x, h_approx)
    f0 = fun(x)

    best_deriv = np.zeros(npt)
    err_deriv = np.zeros(npt)
    best_h = np.zeros(npt)

    display_caution_message = False
    overestimated_indices = []
    for i, f3_i in enumerate(approx_3d):

        if np.abs(f3_i) < eps: 
            
            display_caution_message = True
            overestimated_indices.append(i)
            
            #If third derivative is smaller than eps_machine, computer won't be able to compute h properly. Instead, use approximate solutions. 
            #This does pretty well, but sometimes underestimates errors by a factor of 10. For that reason, I multiplied the error by 100 to overestimate errors.
            

            h = h_approx
            num_deriv = (fun(x[i] + h) - fun(x[i] - h))/(2*h)
            error = 100*eps**(2/3)

            best_deriv[i], err_deriv[i], best_h[i] = num_deriv, error, h


        else: #If third derivative isn't tiny, compute h, numerical derivative and error according to equations (INSERT HERE) . 
            
            
            h = np.cbrt(eps*f0[i]/f3_i)
            num_deriv = (fun(x[i] + h) - fun(x[i] - h))/(2*h)
            
            if np.abs(num_deriv) < eps: #I.e. when derivative is zero, can't avoid divide by zero error. Take rough estimate instead 
                error = eps**(2/3)
                overestimated_indices.append(i)

            else: 
                error = np.cbrt((eps**2)*(f0[i]**2)*f3_i)/(num_deriv)

            best_deriv[i], err_deriv[i], best_h[i] = num_deriv, error, h


    if display_caution_message:
        #I have also included a print statement when this happens to identify which x-values have their errors overstimated.
        print('NOTE: You have derivatives whose errors may be overestimated. Some third derivatives are very close to 0 and so a precise calculation of dx cannot be carried out. See returned indices to see which x-vals are affected. \n'.format(x[i], i)) 


    if full: 

        return best_deriv, err_deriv, best_h, overestimated_indices

    else: 

        return best_deriv


def calculate_third_derivative(fun, x, h):

    #Define function evalulations to compute approximate third derivative.
    #See equation (13) in "Problem Set 1 - Mattias Lazda" (PDF)

    f1 = fun(x + 2*h)
    f2 = -2*fun(x + h)
    f3 = 2*fun(x - h)
    f4 = -fun(x - 2*h)

    return (f1 + f2 + f3 + f4)/(2*h**3)


#Define a function to test
def fun(x):
    return np.cos(x)



x = np.linspace(0, 2*np.pi, 1000)
derivs, errors, h, indices  = ndiff(fun, x , full = True)
#derivs = ndiff(fun, x, full = False)





