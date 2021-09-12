import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

plt.style.use('seaborn-pastel')

def error_cubic_interpolation(fun, x, plot = False):

    y = fun(x)
    dx = x[1] - x[0]

    xx = np.linspace(x[2], x[-3], 1001)
    y_true = fun(xx)
    yy = np.empty(len(xx))

    for i, xx_i in enumerate(xx):
        ind = (xx_i - x[0])/dx
        ind = int(np.floor(ind))
        x_use = x[ind - 1: ind + 3]
        y_use = y[ind - 1: ind + 3]

        p = np.polyfit(x_use, y_use, 3)
        yy[i] = np.polyval(p, xx_i)

    if plot: 
        plt.clf()
        plt.plot(xx, yy, label = 'Cubic Interpolation')
        plt.plot(xx, y_true, label = 'True Function')
        plt.legend()
        plt.show()

    err_cubic = np.std(yy - y_true)

    return err_cubic

def error_spline(fun, x, plot = False):

    y = fun(x)
    xx = np.linspace(x[2], x[-3], 1001)
    y_true = fun(xx)


    spln = interpolate.splrep(x,y)
    yy = interpolate.splev(xx, spln)

    if plot: 
        plt.clf()
        plt.plot(xx, yy, label = 'Cubic Spline')
        plt.plot(xx, y_true, label = 'True Function')
        plt.legend()
        plt.show()

    err_spline = np.std(yy - y_true)

    return err_spline




#Rational Fits are a bit more complicated, so I broke it up into steps

def rat_eval(p,q,x):
    num = 0
    for i,p_i in enumerate(p):
        num += p_i*x**i

    denom = 1
    for j, q_j in enumerate(q):
        denom+=q_j*x**(j+1)

    return num/denom

def rat_fit(x, fun, n, m):
    y = fun(x)
    assert(len(x)==n+m-1)

    mat = np.zeros([n+m-1, n+m-1])

    for i in range(n):
        mat[:,i]=x**i 

    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i

    pars = np.linalg.inv(mat)@y
    p = pars[:n]
    q = pars[n:]

    return p,q


def error_rational(fun, x, n, m, plot = False):

    p,q = rat_fit(x, fun, n, m)
    y = fun(x)
    xx = np.linspace(x[2], x[-3], 1001)
    y_true = fun(xx)
    pred = rat_eval(p,q,xx)

    if plot: 
        plt.clf()
        plt.plot(xx, pred, label = 'Rational Fit')
        plt.plot(xx, y_true, label = 'True Function')
        plt.legend()
        plt.show()
    
    err_rational = np.std(pred - y_true)

    return err_rational



def compare_accuracy(fun, x, n, m, plot = False):

    err_cubic = error_cubic_interpolation(fun, x, plot)
    err_spline = error_spline(fun,x, plot )
    err_rational = error_rational(fun,x,n,m, plot)

    print('''Errors:
-------
Cubic Interpolation: {0}
Cubic Spline: {1}
Rational Fit: {2}
'''.format(err_cubic, err_spline, err_rational))

    return






#Cosine function
n = 3
m = 4
x = np.linspace(-np.pi/2, np.pi/2, n + m - 1)
compare_accuracy(np.cos, x, n, m)


#For lorentzian
def lorentzian(x):
    return 1/(1 + x**2)

n = 4
m = 3
x = np.linspace(-1, 1, n + m - 1)
compare_accuracy(lorentzian, x, n, m, plot = True)
