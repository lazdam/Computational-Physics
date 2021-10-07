import numpy as np

#Define function to integrate. Constant throughout the problem. 

def fun(x,y):
    return y/(1 + x**2)


def rk4_step(fun, x, y, h):

    #Copied from class

    k1 = h*fun(x,y)
    k2 = h*fun(x+h/2, y+k1/2)
    k3 = h*fun(x+h/2, y+k2/2)
    k4 = h*fun(x+h, y+k3)
    dy = (k1+ 2*k2 + 2*k3 + k4)/6
 
    return y + dy

def rk4_stepd(fun, x, y, h):

    #Hardwired to be independent of rk4_step. Guarantees to only have 11 function calls. See equations (1)-(4) in attached PDF. 


    #Calculate y1
    k1 = h*fun(x,y)
    k2 = h*fun(x+h/2, y+k1/2)
    k3 = h*fun(x+h/2, y+k2/2)
    k4 = h*fun(x+h, y+k3)
    dy = (k1+ 2*k2 + 2*k3 + k4)/6

    #Current number of function calls: 4

    y1 = y + dy

    #Calculate y2:

    #1. Calculate first half step
    a = h/2 #Re-scale h
    k1 = k1/2 #This is where we save a function call. 
    k2 = a*fun(x+a/2, y+k1/2)
    k3 = a*fun(x+a/2, y+k2/2)
    k4 = a*fun(x+a, y+k3)
    dy = (k1+ 2*k2 + 2*k3 + k4)/6

    #Current number of function calls: 7

    y2_halfway = y + dy

    #Shift starting variables
    x = x + h/2
    y = y2_halfway

    #2. Calculate second half step
    k1 = a*fun(x, y)
    k2 = a*fun(x+a/2, y+k1/2)
    k3 = a*fun(x+a/2, y+k2/2)
    k4 = a*fun(x+a, y+k3)
    dy = (k1+ 2*k2 + 2*k3 + k4)/6

    #Current number of function calls: 11
    y2 = y + dy

    #Remove leading order according to equation (3) in attached PDF
    Delta = y2 - y1 

    return y2 + Delta/15 #equation (4) in attached PDF



#PART 1:
#--------------------------- 
x = np.linspace(-20, 20, 200)
h = np.median(np.diff(x))
y = np.zeros(len(x))
y[0] = 1 #Set initial value

for i in range(len(x) - 1):
    y[i+1] = rk4_step(fun, x[i], y[i], h)

y_true = np.exp(np.arctan(x) + np.arctan(20)) #True solution to calculate error
error_v1 = np.abs(np.std(y - y_true))

#PART 2: 
#---------------------------
x = np.linspace(-20,20,72)
h = np.median(np.diff(x))
y = np.zeros(len(x))
y[0] = 1

for i in range(len(x) - 1):
    y[i+1] = rk4_stepd(fun, x[i], y[i], h)

y_true = np.exp(np.arctan(x) + np.arctan(20)) #True solution to calculate error
error_v2 = np.abs(np.std(y - y_true))

print('''
PART 1 (rk4_step): Using 200 points (796 function evaluations total), ERROR: {0}
\n
PART 2 (rk4_stepd): Using 72 points (792 function evaluations total), ERROR: {1}
    '''.format(error_v1, error_v2))
