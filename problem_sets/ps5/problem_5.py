import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')

#------------#
#   PART C   #
#------------#
def my_analytic_fft(k, N):
    '''
    Parameters: 
    -----------
    k: float
        Non-integer wavenumber. 
    N: int
        Number of points. 

    Returns: 
        Analytic Fourier Transform of a sin wave with non-integer wavenumber. 
    '''

    my_fft = np.zeros(N, dtype = 'complex')

    for i in range(0,N):
        # Compute analytic solution according to Equation provided in attached PDF
        my_fft[i] = ((1 - np.exp(-2J*np.pi*(i - k)))/(1 - np.exp(-2J*np.pi*(i - k)/N)) - (1 - np.exp(-2J*np.pi*(i + k)))/(1 - np.exp(-2J*np.pi*(i + k)/N)))/2J
        #my_fft[i] = (1 - np.exp(-2J*np.pi*(k - i)))/(1 - np.exp(-2J*np.pi*(k - i)/N))/2J
    return my_fft

# Set number of points
N = 1000

# Define sin function with non-integer k
x = np.arange(N)
k = np.pi
y = np.sin(2*np.pi*k*x/N)

# Get analytic Fourier transform
fft_analytic = my_analytic_fft(k, N)
fft_analytic = np.abs(fft_analytic)

# Get numpy fft
fft_numpy = np.fft.fft(y)
fft_numpy = np.abs(fft_numpy)

# Plot
plt.plot(fft_analytic, label = 'Analytic solution')
plt.plot(fft_numpy, label = 'FFT numpy')
plt.legend()
plt.savefig('figures/analytic_fft_vs_numpy.png')
plt.show()
plt.clf()

# Plot residuals 
plt.plot(fft_analytic - fft_numpy, label = 'residuals')
plt.legend()
plt.savefig('figures/resid.png')
plt.show()

# Calculate error
diff = np.std(fft_analytic - fft_numpy)
print('Error: {0}'.format(diff))

# Sin wave as delta function
k = 1
y = np.sin(k*x)

# Get analytic Fourier transform
fft_analytic = my_analytic_fft(k, N)
fft_analytic = np.abs(fft_analytic)

# Get numpy fft
fft_numpy = np.fft.fft(y)
fft_numpy = np.abs(fft_numpy)

# Plot
plt.plot(fft_analytic, label = 'Analytic solution')
plt.plot(fft_numpy, label = 'FFT numpy')
plt.legend()
plt.savefig('figures/delta_fnc_with_sin.png')
plt.show()
plt.clf()