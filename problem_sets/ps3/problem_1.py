import numpy as np



#Define function to integrate. Constant throughout problem. 
def fun(x,y):
    return y/(1+x**2)

#PART 1: Integrate Using RK4, 4 function evaluations per step
def rk4_step(fun,x,y,h):

    global counter_1
    counter_1+=4 

    k1 = h*fun(x,y)
    k2 = h*fun(x+h/2, y+k1/2)
    k3 = h*fun(x+h/2, y+k2/2)
    k4 = h*fun(x+h, y+k3)
    dy = (k1+ 2*k2 + 2*k3 + k4)/6
 
    return y + dy

#Specify initial value
y0 = 1 
counter_1 = 0

#Specify x values and step size
x = np.linspace(-20,20,100)
h = np.median(np.diff(x))

#Initialize y values. Calculated using rk4_step
y = np.zeros(len(x))
y[0] = y0

for i in range(len(x) - 1):
    y[i+1] = rk4_step(fun, x[i],y[i],h)

y_true = np.exp(np.arctan(x) + np.arctan(20)) #True solution to calculate error
error = np.abs(np.std(y - y_true))

print('Using standard rk4, with a total of {0} function calls, the error is {1}'.format(counter_1, error))


#PART 2: Adaptive RK4

def rk4_stepd(fun,x,y,h):

    left = rk4_step(fun, x, y, h/2)
    right = rk4_step(fun,x + h/2, left, h/2)

    y2 = right
    y1 = rk4_step(fun, x, y, h)

    Delta = y2 - y1

    return y2 + Delta/15


#Specify initial value
y0 = 1 
counter_1 = 0
#Specify x values and step size
x = np.linspace(-20,20,100)
h = np.median(np.diff(x))

#Initialize y values. Calculated using rk4_step
y = np.zeros(len(x))
y[0] = y0

for i in range(len(x) - 1):
    y[i+1] = rk4_stepd(fun, x[i],y[i],h)

y_true = np.exp(np.arctan(x) + np.arctan(20)) #True solution to calculate error
error = np.abs(np.std(y - y_true))

print('Using adaptive rk4, with a total of {0} function calls, the error is {1}'.format(counter_1, error))

















