import numpy as np

def ndiff(fun, x, full = False):
    '''
    Computes numerical derivative of function.

    Parameters:
    -----------
    fun: Function (callable)
        Function you'd like to evaluate the derivative of. 
    x: ndarray
        Array of x values to evaluate numerical derivative at. 
    full: Bool
        Default set to False. If True, function returns derivative, dx and an estimate of error on the derivative. Else, returns derivative only. 

    '''
    
    eps = 2**-52 #default for 64-bit machine
    
    #Determining h. This requires an estimate on the third derivative at x. 
    #An expression for the third derivative is given by equation (11) in "Problem Set 1 - Mattias Lazda" (PDF).

    h_approx = eps**(1/3) #Rough estimate to compute best estimate

    f1 = fun(x + 2*h_approx)
    f2 = -2*fun(x + h_approx)
    f3 = 2*fun(x - h_approx)
    f4 = -fun(x - 2*h_approx)


    approx_3_deriv = (f1 + f2 + f3 + f4)/(2*h_approx**3)
    approx_3_deriv = max(approx_3_deriv, eps) #When third derivative is zero, choose smallest possible value.

    h_best = ((fun(x)*eps)/approx_3_deriv)**(1/3) #Computes h according to equation (5.7.8) in Numerical Recipes

    
    deriv = (fun(x + h_best) - fun(x - h_best))/(2*h_best) #Compute double-sided derivative

    print('Fractional error: ', deriv/-np.sin(x) - 1)

    if full:

        f = fun(x)

        err_deriv = (eps**(2/3))*(f**(2/3))*(approx_3_deriv**(1/3))/deriv #Fractional error according to equation (5.7.9) in Numerical Recipes.
        
        
        return deriv, h_best, err_deriv



    return deriv


#Define a function to test
def fun(x):
	return np.cos(x)


#Main function that computes and prints desired results. 
def main(fun, x, full):

    if full:
        num_deriv = ndiff(fun, x, full)

        print('''
        Numerical Derivatives: 
        ----------------------
        x = {0}
        f'(x) = {1}
        err_deriv = {2}
        h = {3}\n'''.format(x, num_deriv[0], num_deriv[2], num_deriv[1]))
	
    else: 
        num_deriv = ndiff(fun, x, full)

        print('''
        Numerical Derivatives:
        ----------------------
        x = {0} 
        f'(x) = {1}\n'''.format(x, num_deriv))

	    




x = np.array([np.pi/4, np.pi/3]) #Why no work for x = 20? 
full_arr = [True]

if __name__ == "__main__":
	
    for full in full_arr:

	    main(fun, x, full)


