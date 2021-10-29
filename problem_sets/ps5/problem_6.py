import numpy as np
import matplotlib.pyplot as plt

n = 1000
rw = np.cumsum(np.random.randn(n))


ft = np.fft.rfft(rw)

plt.plot(np.abs(ft[1:]))
plt.plot(np.linspace(0,1000,1000)**(-2))
plt.show()