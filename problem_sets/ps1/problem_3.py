import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def lakeshore(V, data):
    '''
    Provides interpolated temperature given some input voltage. 

    Parameters: 
    -----------

    V: float or ndarray
        Input voltage
    data: '.txt' file
        Textfile containing information about the diodes. First column
        corresponds to temperature, second column corresponding voltage
        and third column dV/dT. 

    Returns: ndarray
        Interpolated temperature
    '''
    T_raw, V_raw, dVdT = np.loadtxt('lakeshore.txt').T

        
    interpolated_temps = np.array([])
    interpolated_errors = np.array([])


    #If data isn't in an array, put it into one. 
    if type(V) == type(np.array([])):
        pass
    else: 
        V = np.array([V])


    for i, V_input in enumerate(V): 

        indices = get_nearest_neighbor(V_raw, V_input) #Get indices of 4 nearest neighbors
        x_use = V_raw[indices[0][1:]] #[0] index to get nearest neighbors of V, [1:] to exclude index of itself as NN. 
        y_use = T_raw[indices[0][1:]] #Get corresponding temperatures of nearest voltages. 

        x_use =np.sort(x_use) #Sort in increasing order
        y_use = np.sort(y_use)[::-1] #Utilizing the fact that y-vals are decreasing

        #Interpolate, cubic polynomial
        p = np.polyfit(x_use, y_use, 3) #Fit cubic polynomial to 4 points
        interp_temp = np.polyval(p, V_input) #Calculate interpolated temperature

        #Estimate error
        err_interp = estimate_error(V_input,interp_temp, x_use, y_use)

        #Append to arrays
        interpolated_temps = np.append(interpolated_temps, interp_temp)
        interpolated_errors = np.append(interpolated_errors, err_interp)
            

    return interpolated_temps, interpolated_errors


def get_nearest_neighbor(V_raw, V):
    '''
    Determines Nearest Neighbours of V using sklearn.

    Parameters:
    -----------
    V_raw: ndarray
        Raw input voltages
    V: float
        Desired input voltage

    Returns: ndarray
        Indices of 4 nearest neighbors of V
    '''

    V_temp = V_raw.reshape(-1,1)
    V_neighbors = np.vstack((V, V_temp)) #Add input voltage to raw voltages
    
    #Find the four nearest neighbors closest to inputted voltage using NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=5, algorithm = 'ball_tree').fit(V_neighbors) #includes itself as neighbor, so 1 + 4 = 5 nearest neighbors asked for
    distances, indices = nbrs.kneighbors(V_neighbors)

    return indices

def estimate_error(V, T_interp, V_neighbor, T_neighbor):
    '''
    Estimates the uncertainty in interpolated temperature. Supporting theory can be found in Section 3.1. 

    Parameters:
    -----------
    V: float or ndarray
        Inputted voltage used to determine interpolated temperature. 
    T_interp: float or ndarray
        Interpolated temperature corresponding to V
    V_neighbor: ndarray
        4 Nearest neighbors of V. Must be in increasing order.
    T_neighbor: ndarray
        4 temperatures corresponding to V_neighbor. Must align with ordering of V_neighbor

    Returns: ndarray

    '''
    vv = np.linspace(min(V_neighbor), max(V_neighbor), 1001)
    max_g = np.abs(np.max((vv - V_neighbor[0])*(vv - V_neighbor[1])*(vv - V_neighbor[2])*(vv - V_neighbor[3]))) #See equation (19) in PDF

    T0,T1,T2,T3 = T_neighbor[0],T_neighbor[1],T_neighbor[2], T_neighbor[3]

    max_fourth_deriv = np.abs((T0 - 4*T1 + 6*T_interp - 4*T2 + T3)/(np.min(V - V_neighbor))**4) #See equation (18) in attached PDF
    
    error = max_g*max_fourth_deriv/24 #See equation (21) in attached PDF

    return error









V = 0.5
#V = np.linspace(0.2, 1.6, 5)

data = np.loadtxt('lakeshore.txt')
temps, errors = lakeshore(V, data)

print('''
Input Voltage(s):
---------------
V = {0}V

Interpolated Temperature(s):
--------------------------
T = {1}K

Estimated Error(s) on Interpolated Temperature(s):
---------------------------------------------
err_T = {2}K

    '''.format(V, temps, errors))

