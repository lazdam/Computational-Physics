import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

plt.style.use('seaborn-pastel')



#Define constants and variables: 
R = 1.0
sigma = 1
epsilon = 1
z = np.linspace(0, 20, 1001) #Includes R = 1!
npts = len(z)


def func(theta, z):
    '''
    Computes the integrand to compute the electric field of a spherical shell. 

    Parameters: 
    -----------
    theta: float
        angle of integration e [0,pi].
    z: float/ndarray
        distances to compute electric field at. 

    Returns: float/ndarray
        Integrand
    '''

    global sigma
    global epsilon
    global R

    #See equation (1) in attached PDF
    const = (R**2)*sigma/2/epsilon
    numerator = (z - R*np.cos(theta))*np.sin(theta)
    denominator = (R**2 + z**2 - 2*R*z*np.cos(theta))**(3/2)
    
    return const*numerator/denominator


def integrate_adaptive(func, theta0, theta1, z, tol):
    '''
    Adapative integration of an electric field using Simpson's rule. 

    Parameters: 
    -----------
    func: function, callable
        Function you'd like to integrate. 
    theta0: float
        Lower bound of integration.
    theta1: float
        Upper bound of integration.
    z: float
        Distance away from surface of shell. 
    tol: float
        error tolerance

    Returns: float
        area/integral. In this case, the electric field. 

    '''
    
    #Below is simply Jon's code with modified variable names to fit the problem. 
    theta = np.linspace(theta0, theta1, 5) 
    y = func(theta, z)
    
    dtheta = (theta1 - theta0)/(len(theta) - 1)

    area1 = 2*dtheta*(y[0]+4*y[2]+y[4])/3
    area2=dtheta*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3

    err = np.abs(area1-area2)

    if err<tol:
        return area2

    else: 
        theta_mid = (theta0+theta1)/2
        left=integrate_adaptive(func,theta0,theta_mid, z, tol/2)
        right=integrate_adaptive(func,theta_mid, theta1, z,tol/2)
        return left+right


#Initialize electric field arrays
E_fields = np.zeros(npts)
E_fields_quad = np.zeros(npts)

#Set the error tolerance
tol = 1e-7

for i in range(npts):
 
    #Since the integral is dependent on z, define a temporary function
    #which takes into account the current z value so we can integrate
    #in a loop. 
    def func_temp(theta):
        return func(theta, z[i])
    
    #Use quad to compute integral for current function at z_i
    E_i_quad = integrate.quad(func_temp, 0, np.pi)
    E_fields_quad[i] = E_i_quad[0]
    
    #I noticed that integrate_adaptive would run into a recursion error. 
    #Skipped the integration at z = R. 
    if z[i] == R:
        print('Singularity at z = R. Unable to integrate due to RecursionError. Default integral set to 0.')
        continue
    
    E_i = integrate_adaptive(func, 0, np.pi, z[i], tol)
    E_fields[i] = E_i
    


if False: 

    #Lastly, I just plotted some results to see what it would look like. See Figure 1 in attached PDF. 

    z_true = np.linspace(R,20,101)
    E_true = 1/z_true**2

    plt.plot(z, E_fields, label = 'My integral')
    plt.plot(z, E_fields_quad, ls = '--', label = 'quad')
    plt.plot(z_true, E_true, label = 'True field')
    plt.xlabel('z [m]')
    plt.ylabel('E [V/m]')
    plt.legend()
    plt.show()
