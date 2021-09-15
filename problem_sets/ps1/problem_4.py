import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def error_cubic_interpolation(fun, x):

    y = fun(x)
    dx = x[1] - x[0]

    xx = np.linspace(x[2], x[-3], 1001) #Since cubic interpolation needs 4 nearest neighbors, can't use first and last two endpoints 
    y_true = fun(xx)
    yy = np.empty(len(xx))

    for i, xx_i in enumerate(xx):
        ind = (xx_i - x[0])/dx
        ind = int(np.floor(ind))
        x_use = x[ind - 1: ind + 3]
        y_use = y[ind - 1: ind + 3]

        p = np.polyfit(x_use, y_use, 3)
        yy[i] = np.polyval(p, xx_i)

    err_cubic = np.std(yy - y_true)

    return err_cubic

def error_spline(fun, x):

    y = fun(x)
    xx = np.linspace(x[2], x[-3], 1001)
    y_true = fun(xx)


    spln = interpolate.splrep(x,y)
    yy = interpolate.splev(xx, spln)


    err_spline = np.std(yy - y_true)

    return err_spline


#Rational Fits are a bit more complicated, so I broke it up into steps

def rat_eval(x, p, q):

    #Recall p is polynomial of degree n, with n+1 coeffs (p = a_0 + a_1x^1 + ... + a_nx^n). 
    #q is polynomial of degree m with m coeffs (q = b_1x^1 + b_2x^2 + ... + b_mx^m)
    
    top = 0
    for i, p_i in enumerate(p):
        top+=p_i*x**i
    
    bot = 1    
    for j, q_j in enumerate(q):
        
        bot+=q_j*x**(j+1)
    
    return top/bot

def rat_fit(x, fun, n, m, pinv = False): #n and m represent degree of polynomial, not number of terms
    
    assert(len(x)==n+m+1)

    y = fun(x)
    
    mat = np.zeros([n+1+m, n+1+m])
    
    for i in range(n+1):
        mat[:,i] = x**i

    for j in range(0,m):
        mat[:,n+1+j] = -y*x**(j+1)
        

    if pinv: 

        coeffs = np.linalg.pinv(mat)@y

    else: 

        coeffs = np.linalg.inv(mat)@y
    
    p = coeffs[:n+1]
    q = coeffs[n+1:]
    
    return p, q




def error_rational(fun, x, n, m, pinv = False):

    p,q = rat_fit(x, fun, n, m, pinv)
    y = fun(x)
    xx = np.linspace(x[2], x[-3], 100) #compare std deviation on same interval as cubic poly and spline
    y_true = fun(xx)
    
    yy = np.empty(len(xx))
    for i, xx_i in enumerate(xx):
        yy_i = rat_eval(xx_i, p, q)
        yy[i] = yy_i

    err_rational = np.std(yy - y_true)

    return err_rational



def compare_accuracy(fun, x, n, m, pinv = False):

    err_cubic = error_cubic_interpolation(fun, x)
    err_spline = error_spline(fun,x )
    err_rational = error_rational(fun,x,n,m, pinv)

    print('''Errors:
-------
Cubic Interpolation: {0}
Cubic Spline: {1}
Rational Fit: {2}
------------------------------------------\n'''.format(err_cubic, err_spline, err_rational))

    return



#BROKEN DOWN INTO MULTIPLE SECTIONS TO ANSWER EACH QUESTION

#a) Cosine function
n = 4
m = 5
x = np.linspace(-np.pi/2, np.pi/2, n + m + 1)
print('a) Comparing Accuracy for f(x) = cos(x). m = {0} & n = {1}'.format(n,m))
compare_accuracy(np.cos, x, n, m)


#b) Lorentzian function. pinv = False (using np.linalg.inv)
def lorentzian(x):
    return 1/(1 + (x)**2)

n = 4
m = 5
x = np.linspace(-1, 1, n + m + 1)
print('b) Comparing Accuracy for f(x) = 1/(1 + x^2). m = {0} & n = {1}. Note: np.linalg.inv used.'.format(n,m))

compare_accuracy(lorentzian, x, n, m)

#c) Lorentzian function. pinv = True
print('c) Comparing Accuracy for f(x) = 1/(1 + x^2). m = {0} & n = {1}. Note: np.linalg.pinv used.'.format(n,m))
compare_accuracy(lorentzian, x, n, m, pinv = True)


#d) Comparing p, q from np.linalg.inv vs. np.linalg.pinv
p_old, q_old = rat_fit(x, lorentzian, n, m , pinv = False)
p_new, q_new = rat_fit(x, lorentzian, n, m, pinv = True)

print('''d) Comparing coefficients before and after using pinv
p_old = {0}
q_old = {1}
p_new = {2}
q_new = {3}

    '''.format(p_old, q_old, p_new, q_new))



