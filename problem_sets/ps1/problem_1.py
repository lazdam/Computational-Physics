import numpy as np 

#Part b): 

#NOTE: For the sake of this code, I have re-labelled \delta as h.

eps_machine = 2**-52 #default for 64-bit computers


def calculate_first_derivative(fun, x, h):
    '''
    Computes approximate first derivative using centered difference approximation (see equation (7) in attached PDF). 

    Parameters: 
    -----------
    fun: function
        Input function
    x: ndarray/float
        x values to evaluate derivative at
    h: step size

    Returns: ndarray/float
        derivative
    '''

    f1 = 8*fun(x + h)
    f2 = -8*fun(x - h)
    f3 = -1*fun(x + 2*h)
    f4 = fun(x - 2*h)

    deriv = (f1 + f2 + f3 + f4)/(12*h)
    
    return deriv


#1) f(x) = exp(x)
def fun(x):
    return np.exp(x)

x = 42
h = eps_machine**(1/5) #Since f/f^(5) = 1 


deriv = calculate_first_derivative(fun, x, h)

fractional_err_predicted = eps_machine**(4/5) #Fractional accuracy

print('''
f(x) = exp(x):
--------------

Derivative at x = {0} is {1} with true fractional error {2}.
The predicted fractional accuracy of the computed derivative is proportional to {3}\n
    '''.format(x, deriv, deriv/fun(x) - 1, fractional_err_predicted))


#2) f(x) = exp(0.01x)
def fun(x):
    return np.exp(0.01*x)

h_new = (eps_machine**(1/5))*100 #since (f/f^(5))^(1/5) = (f/((0.01^5)*(f))^(1/5) = 1/0.01 = 100
x = 4200

deriv = calculate_first_derivative(fun, x, h_new)
fractional_err_predicted = eps_machine**(4/5) #Remains the same, all other terms cancel each other out.

print('''
f(x) = exp(0.01x):
------------------

Derivative at x = {0} is {1} with true fractional error {2}.
The predicted fractional accuracy of the computed derivative is proportional to {3}\n
    '''.format(x, deriv, deriv/(0.01*fun(x)) - 1, fractional_err_predicted))