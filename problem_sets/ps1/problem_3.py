import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

plt.style.use('seaborn-pastel')

def lakeshore(V, data):
    '''
    Provides interpolated temperature given some input voltage. 

    Parameters: 
    -----------

    V: float
        Input voltage
    data: '.txt' file
        Textfile containing information about the diodes. First column
        corresponds to temperature, second column corresponding voltage
        and third column dV/dT. 

    Returns: float
        Interpolated temperature
    '''
    T_raw, V_raw, dVdT = np.loadtxt('lakeshore.txt').T

        
    interpolated_temps = np.array([])

    if type(V) == type(np.array([])):

        for i, V_input in enumerate(V):

            indices = get_nearest_neighbor(V_raw, V_input)
            x_use = V_raw[indices[0][1:]] #[0] index to get nearest neighbors of V, [1:] to exclude index of itself as NN. 
            y_use = T_raw[indices[0][1:]]

            p = np.polyfit(x_use, y_use, 3) #Fit cubic polynomial to 4 points
            interp_temp = np.polyval(p, V_input) #Calculate interpolated temperature
            interpolated_temps = np.append(interpolated_temps, interp_temp)
            

    else: 

        indices = get_nearest_neighbor(V_raw, V)
        x_use = V_raw[indices[0][1:]]
        y_use = T_raw[indices[0][1:]]

        p = np.polyfit(x_use, y_use, 3) 
        interp_temp = np.polyval(p, V) 
        interpolated_temps = np.append(interpolated_temps, interp_temp)
        



    return interpolated_temps


def get_nearest_neighbor(V_raw, V):

    V_temp = V_raw.reshape(-1,1)
    V_neighbors = np.vstack((V, V_temp)) #Add input voltage to raw voltages
    
    #Find the four nearest neighbors closest to inputted voltage using NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=5, algorithm = 'ball_tree').fit(V_neighbors)
    distances, indices = nbrs.kneighbors(V_neighbors)

    return indices





V = 1.5
data = np.loadtxt('lakeshore.txt')
print(lakeshore(V, data))
