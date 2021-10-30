import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-muted')

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
    
    return my_fft

# Set number of points
N = 100

# Define sin function with non-integer k
x = np.arange(N)
k = 25.55
y = np.sin(2*np.pi*k*x/N)

# Get analytic fourier transform
fft_analytic = np.abs(my_analytic_fft(k, N))
rfft_analytic = fft_analytic[0 : N//2 + 1]
# Get numpy fft
fft_numpy = np.abs(np.fft.rfft(y))

# Plot results
plt.plot(fft_numpy[:], label = 'rFFT')
plt.plot(rfft_analytic[:], label = 'Real Analytic FT', ls = '--')
plt.xlabel('k')
plt.legend()
plt.savefig('figures/analytic_vs_fft.png')
plt.show()


# Plot Residuals
plt.plot((fft_numpy - rfft_analytic)[:])
plt.xlabel('k')
plt.savefig('figures/ft_comparison_residuals.png')
plt.show()
# Get error
print('Error:{0}'.format(np.abs(np.std((fft_numpy - rfft_analytic)[:]))))


#------------#
#   PART D   #
#------------#
# Set number of points
N = 100

# Define sin function with non-integer k
x = np.arange(N)
k = 25.55
y = np.sin(2*np.pi*k*x/N)
window = 0.5 - 0.5*np.cos(2*np.pi*x/N)
fft_windowed = np.abs(np.fft.rfft(y*window))

plt.plot(fft_numpy, label = 'rFFT')
plt.plot(fft_windowed, label = 'Windowed rFFT')
plt.legend()
plt.xlabel('k')
plt.savefig('figures/windowed_fft.png')
plt.show()

#------------#
#   PART E   #
#------------#

# Get FT of the window function
fft_window = np.fft.fft(window)
# Generate expected result
expected = np.zeros(len(fft_window))
expected[0] = N/2
expected[1] = -N/4
expected[-1] = -N/4

# Plot and compare
plt.plot((fft_window), label = 'FFT of window')
plt.plot(expected, label = 'Expected', ls = '--')
plt.legend()
plt.xlabel('k')
plt.savefig('figures/fft_window_q5e.png')
plt.show()

plt.plot((fft_window)- expected)
plt.xlabel('k')
plt.savefig('figures/fft_window_q5e_resid.png')
plt.show()

# Get smoothed FFT
yft = np.fft.rfft(y)
yft_smooth = 0.5*yft - 0.25*np.roll(yft,1) - 0.25*np.roll(yft, -1)

# Compare to previously obtained windowed FFT
plt.plot(np.abs(yft_smooth), label = 'smooth FT')
plt.plot(fft_windowed, label = 'Windowed fft', ls = '--')
plt.legend()
plt.xlabel('k')
plt.savefig('figures/smooth_fft_vs_windowed.png')
plt.show()

