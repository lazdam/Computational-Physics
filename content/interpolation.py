import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-pastel')

#Quadratic Interpolation
x = np.linspace(-1,1,5)
y = np.abs(x)

xx = np.linspace(-1.2,1.2,1000)

plt.clf()

plt.plot(x,y,'*', label = 'f(x) = |x|')

for i in range(len(x) - 2):
    pp = np.polyfit(x[i:i+3], y[i:i+3],2)
    yy = np.polyval(pp,xx)
    plt.plot(xx,yy, label = 'fit{0}'.format(i))
    plt.title('Quadratic Interpolation')

# plt.legend()
# plt.savefig('quadratic_interpolation_example')
# plt.show()

plt.clf()


#Cubic Interpolation
x = np.linspace(-1,1,6)
y = np.abs(x)
plt.plot(x,y,'*', label = 'f(x) = |x|')


xx = np.linspace(-1.2,1.2,1000)

for i in range(len(x) - 3):
    pp = np.polyfit(x[i:i+4], y[i:i+4],3)
    yy = np.polyval(pp,xx)
    plt.plot(xx,yy, label = 'fit{0}'.format(i))
    plt.title('Cubic Interpolation')

# plt.legend()
# plt.savefig('cubic_interpolation_example')
# plt.show()

plt.clf()

#Another example
import numpy as np
xi = np.linspace(-2,2,11)
yi = np.exp(xi)

yi[-2:] = y[-3]

x = np.linspace(xi[1], xi[-3], 1001)
y_true = np.exp(x)
y_interp=np.zeros(len(x))

for i in range(len(x)):
    ind=np.max(np.where(x[i]>=xi)[0])
    x_use=xi[ind-1:ind+3]
    y_use=yi[ind-1:ind+3]
    pars=np.polyfit(x_use,y_use,3)
    pred=np.polyval(pars,x[i])
    y_interp[i]=pred


plt.plot(x,y_interp)
plt.plot(xi,yi,'--')
plt.show()


