import numpy as np
import matplotlib.pyplot as plt
def ndiff(x, fun, full = False):

    eps = 2**-52

    h_est = eps**(1/3)

    f0 = fun(x)
    f1 = fun(x + 2*h_est)
    f2 = -1*fun(x - 2*h_est)
    f3 = -2*fun(x + h_est)
    f4 = 2*fun(x - h_est)

    approx_3d = (f1 + f2 + f3 + f4)/(2*h_est**3)
    

    print(np.absolute(approx_3d) < eps)
    if np.abs(approx_3d) < eps:
        
        num_deriv = (fun(x + h_est) - fun(x - h_est))/(2*h_est)
        err_deriv = eps**(2/3)
        h = h_est


        print('done')

    else: 
        h = np.cbrt(eps*f0/approx_3d)
        num_deriv = (fun(x + h) - fun(x - h))/(2*h)
        err_deriv = np.cbrt((eps**2)*(f0**2)*approx_3d)/(num_deriv)

        print('yeet')

    print('Fractional error:', num_deriv/(1/(1 + x**2))- 1)

    if full: 

        return num_deriv, err_deriv, h


    return num_deriv 


def fun(x):
    return np.arctan(x)


x = np.linspace(0,2*np.pi, 100)


for i, xx_i in enumerate(x):

    print(ndiff(xx_i, fun, full = True))
    print('------------------------------\n')

     
    

