import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-muted')

n = 1000

# Generate random walk and get power spectrum
rw = np.cumsum(np.random.randn(n))
rFFT = np.abs(np.fft.rfft(rw))

# Generate rational function
x = np.linspace(1.3, n//2 + 1, 1000)
rat_fun = 5*n*(x - 0.7)**(-2)

# Plot
plt.plot(rFFT[1:], label = 'Power Spectrum')
plt.plot(x, rat_fun , label = 'k$^{-2}$')
plt.legend()
plt.xlabel('k')
plt.savefig('figures/rw_ps.png')
plt.show()