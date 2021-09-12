import numpy as np 

#Part b): 

#NOTE: For the sake of this code, I have re-labelled \delta as h.

eps_machine = 2**-52 #default for 64-bit computers


#1) f(x) = exp(x)
x = 42
h = eps_machine**(1/5) #Since f/f^(5) = 1 

#Compute the derivative according to equation (5) in attached PDF. Four function evals:

f0 = np.exp(x)
f1 = 8*np.exp(x + h)
f2 = -8*np.exp(x - h)
f3 = -1*np.exp(x + 2*h)
f4 = np.exp(x - 2*h)

deriv = (f1 + f2 + f3 + f4)/(12*h)
fractional_err_theory = eps_machine**(4/5)

print('''
f(x) = exp(x):
--------------

Derivative at x = {0} is {1} with fractional error {2}.
The theoretical fractional accuracy of the computed derivative is proportional to {3}\n
    '''.format(x, deriv, np.exp(x)/deriv - 1, fractional_err_theory))


#2) f(x) = exp(0.01x)

h_new = (eps_machine**(1/5))*100 #since (f/f^(5))^(1/5) = (f/((0.01^5)*(f))^(1/5) = 1/0.01 = 100
x = 4200 #scaled up

#Compute the derivative according to equation (5) in attached PDF. Four function evals:

f0 = 0.01*np.exp(0.01*x) #True derivative

f1 = 8*np.exp(0.01*(x + h_new))
f2 = -8*np.exp(0.01*(x - h_new))
f3 = -1*np.exp(0.01*(x + 2*h_new))
f4 = np.exp(0.01*(x - 2*h_new))

deriv_new = (f1 + f2 + f3 + f4)/(12*h_new)
fractional_err_theory = eps_machine**(4/5) #stays the same 

print('''
f(x) = exp(0.01x):
------------------

Derivative at x = {0} is {1} with fractional error {2}.
The theoretical fractional accuracy of the computed derivative is proportional to {3}\n
    '''.format(x, deriv_new, f0/deriv_new - 1, fractional_err_theory))