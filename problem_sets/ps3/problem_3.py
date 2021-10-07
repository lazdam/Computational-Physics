import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-pastel')

#load data
data = np.loadtxt('dish_zenith.txt')
x,y,z = data.T
npt = len(x)

#Create A matrix according to equation (8) in attached PDF
A = np.zeros((npt,4))
A[:,0] = x**2 + y**2
A[:,1] = y
A[:,2] = x
A[:,3] = 1

#Decompose A matrix using SVD
U, S, V_T = np.linalg.svd(A, False)

#Invert S
S_inv = np.asarray(np.diag(1.0/S)) #need to put it into a matrix

#Calculate best fit parameters
m = V_T.T@S_inv@U.T@z

a, B, C, D = m

#Find x0, y0, z0 
x0 = C/(-2*a)
y0 = B/(-2*a)
z0 = D - a*x0**2 - a*y0**2 


print('''
[PART B]: 

Best Fit Parameters (before conversion):
---------------------------------------
a: {0} mm-1
B: {1}
C: {2}
D: {3} mm
'''.format(a, B, C, D))

print('''
Best Fit Parameters (after conversion):
---------------------------------------
a: {0} mm-1
x0: {1} mm
y0: {2} mm
z0: {3} mm
'''.format(a, x0, y0, z0))

#Compute residuals and standard deviation of residuals
residuals = A@m - z
std = np.std(residuals)

#Plot histogram to see distribution of residuals
plt.hist(residuals, color = 'red')
plt.ylabel('Number of data points')
plt.xlabel('Δz')
plt.savefig('./figure3.pdf')
plt.show()

#Residuals look normally distributed, so estimate N to be uncorrelated
N = np.eye(npt)*std**2
N_inv = np.linalg.inv(N)

#Estimate error in model parameters
err_params = np.sqrt(np.diag(np.linalg.inv(A.T@N_inv@A)))

#Calculate focal length and error, converted to meters
f = (1/(4*a))/1000
err_f = (f*err_params[0]/a)


print('''
[PART C]: 

The Noise was estimated to be uncorrelated with standard deviation {0}. From this, we estimated the error 
in 'a' according to equation (9) in attached PDF.

Error on a: {1} mm-1

Using equations (10) and (11) in attached PDF, we calculated the focal length to be: 
f = {2}m with error {3}m.

Taking into account significant digits: 
f = {4} +/- {5} m, 
binging us well in range of the desired 1.5m!

    '''.format(std, err_params[0], f, err_f , round(f, 4), round(err_f, 4)))






