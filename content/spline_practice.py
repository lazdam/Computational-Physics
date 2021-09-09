import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-np.pi, np.pi, 20)
y = np.sin(x)
dx = x[1]-x[0]


xx = np.linspace(x[2], x[-3], 1001)
yy = np.empty(len(xx))



for i in range(len(xx)): #iterate through each index
    ind = (xx[i] - x[0])/dx #left neighbour 
    ind = int(np.floor(ind))
    x_use = x[ind - 1: ind + 3]
    y_use = y[ind - 1: ind + 3]
    p = np.polyfit(x_use, y_use, 3)
    yy[i] = np.polyval(p, xx[i])




plt.plot(x,y,'*')
plt.plot(xx, np.sin(xx), '--', label = 'True value')
plt.plot(xx, yy, label = 'cubic interpolation')
plt.legend()

plt.show()