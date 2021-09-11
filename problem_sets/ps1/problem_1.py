import numpy as np 

def numerical_deriv(x,C,ε):
    '''
    Computes the 5-point derivative according to equation (5) (see attached pset1 PDF). 
    
    Parameters:
    -----------
    x: int/float
        x-value where derivative is being evaluated at
    ε_machine: float
        ε of machine.
    
    Returns: Estimated derivative (float)
    '''
    
    #Compute δ (see equation (7))
    δ = ε**(1/5)
    temp = x - δ
    δ = x - temp


    #Compute x vals where f is evaluated at
    deriv = (8*func(C, x + δ) - 8*func(C, x - δ) - func(C, x + 2*δ) + func(C, x - 2*δ))/(8*δ)

    return deriv     

def func(C, x):
    '''
    Evaluates exp(C*x) at x.

    Parameters: 
    -----------
    C: float
        Steepness of exponential function. 
    x: int/float
        Input x value

    Returns: float
    '''    
    return np.exp(C*x) 

#--------------------------------------------#	

ε = 2**-52

#Set x value to evaluate derivative at:
x = 42

#Set array of C values requested in assignment.
C_arr = [1]

deriv_arr = np.empty(len(C_arr)) #initialize empty array to store computed derivatives

for i, C in enumerate(C_arr):
    
    deriv = numerical_deriv(x, C, ε)
    deriv_arr[i] = deriv

#print(deriv_arr)
#print(deriv_arr[0]/np.exp(x) - 1)
#print(ε**(4/5))




