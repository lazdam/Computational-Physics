"""
Problem Set 7, Problem 1
Mattias Lazda
260845451
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# Test C random number generator 
x,y,z = np.loadtxt('rand_points.txt').T

fig = plt.figure()
ax = plt.axes(projection='3d')

# Reduce the number of points to see the lines
n = 2000
ax.scatter(x[:n],y[:n],z[:n], c=z[:n], cmap = 'viridis', linewidth =0.1)
plt.show()
plt.clf()
# Save image after it has been rotated to proper viewing angle. 

# Test numpy random number generator
coords = np.random.randint(1e6, 1e8, size = (3,n))
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter(coords[0], coords[1], coords[2], c = coords[2])
plt.show()
