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


    print(approx_3d < h_est)
    if approx_3d < h_est:
        approx_3d = h_est #Set to small value if third derivative is zero to avoid nan/inf
        h = (eps*f0/approx_3d)**(1/3)
        num_deriv = (fun(x + h) - fun(x - h))/(2*h)
        err_deriv = 1000*eps**(2/3)


        print('done')

    else: 
        h = (eps*f0/approx_3d)**(1/3)
        num_deriv = (fun(x + h) - fun(x - h))/(2*h)
        err_deriv = (((eps**2)*(f0**2)*approx_3d)**(1/3))/(num_deriv)

        print('yeet')

    print('Fractional error:', num_deriv/(5*x**4 + 6*x)- 1)

    if full: 

        return num_deriv, err_deriv, h


    return num_deriv 


def fun(x):
    return x**5 + 3*x**2 + 1

print(ndiff(5, fun, full = True))
x = np.linspace(0, 2*np.pi, 1000)

y = np.empty(len(x))
for i, x_i in enumerate(x):
    y[i] = ndiff(x_i, fun, full = False)

plt.plot(x,y, label = 'derivative')
plt.plot(x, 5*x**4 + 6*x, label = 'true')
plt.legend()
plt.show()

plt.plot(x, y - (5*x**4 + 6*x))
plt.show()
