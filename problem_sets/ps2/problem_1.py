import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#Define constants and variables: 
R = 1.0
sigma = 1
epsilon = 1
z = np.linspace(0, 20, 1001)
print(z)
npts = len(z)


def func(theta, z):

    global sigma
    global epsilon
    global R


    const = (R**2)*sigma/2/epsilon
    numerator = (z - R*np.cos(theta))*np.sin(theta)
    denominator = (R**2 + z**2 - 2*R*z*np.cos(theta))**(3/2)
    
    return const*numerator/denominator


def integrate_adaptive(func, theta0, theta1, z, tol):

    #Hardwire to use Simpsons

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


E_fields = np.zeros(npts)
E_fields_quad = np.zeros(npts)
tol = 1e-7

for i in range(npts):

    def func_temp(theta):
        return func(theta, z[i])

    E_i_quad = integrate.quad(func_temp, 0, np.pi)
    E_fields_quad[i] = E_i_quad[0]

    if z[i] == R:
        print('Singularity at z = R. Skipping to avoid RecursionError.')
        continue
    

    E_i = integrate_adaptive(func, 0, np.pi, z[i], tol)
    E_fields[i] = E_i


if True: 

    z_true = np.linspace(R,20,101)
    E_true = 1/z_true**2

    plt.plot(z, E_fields, label = 'My integral')
    plt.plot(z, E_fields_quad, ls = '--', label = 'quad')
    plt.plot(z_true, E_true, label = 'True field')
    plt.legend()
    plt.savefig('q1')
    plt.show()
