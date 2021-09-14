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
        plt.plot(x, y, "*", label = 'Data')
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
        plt.plot(x, y, "*", label = 'Data')
        plt.plot(xx, y_true, label = 'True Function')
        plt.legend()
        plt.show()

    err_spline = np.std(yy - y_true)

    return err_spline




#Rational Fits are a bit more complicated, so I broke it up into steps

def rat_eval(x, p, q):
    
    top = 0
    for i, p_i in enumerate(p):
        top+=p_i*x**i
    
    bot = 1    
    for j, q_j in enumerate(q):
        
        bot+=q_j*x**(j+1)
    
    return top/bot

def rat_fit(x, fun, n, m): #n and m represent degree of polynomial, not number of terms
    
    assert(len(x)==n+m+1)

    y = fun(x)
    
    mat = np.zeros([n+1+m, n+1+m])
    
    for i in range(n+1):
        mat[:,i] = x**i

    for j in range(0,m):
        mat[:,n+1+j] = -y*x**(j+1)
        
    coeffs = np.linalg.pinv(mat)@y
    
    p = coeffs[:n+1]
    q = coeffs[n+1:]
    
    return p, q




def error_rational(fun, x, n, m, plot = False):

    p,q = rat_fit(x, fun, n, m)
    y = fun(x)
    xx = np.linspace(x[0], x[-1], 100)
    y_true = fun(xx)
    
    yy = np.empty(len(xx))
    for i, xx_i in enumerate(xx):
        yy_i = rat_eval(xx_i, p, q)
        yy[i] = yy_i

    


    if plot: 
        plt.clf()
        plt.plot(xx, yy, label = 'Rational Fit')
        plt.plot(x, y, "*", label = 'Data')
        plt.plot(xx, y_true, label = 'True Function')
        plt.legend()
        plt.savefig('err_rational_lorentzian_n4_m5_pinv')
        plt.show()
    
    err_rational = np.std(yy - y_true)

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
n = 4
m = 5
x = np.linspace(-np.pi/2, np.pi/2, n + m + 1)
#compare_accuracy(np.cos, x, n, m, plot = True)


#For lorentzian
def lorentzian(x):
    return 1/(1 + (x)**2)

n = 4
m = 5
x = np.linspace(-1, 1, n + m + 1)
compare_accuracy(lorentzian, x, n, m, plot = True)
