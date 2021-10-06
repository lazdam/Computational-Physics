import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import astropy.units as un

plt.style.use('seaborn-pastel')



#------------#
#   PART A   #
#------------#

#Store names and half lives. NB: Half lives converted to minutes. 

products = [
            'U238', 'T234', 'P234', 'U234', 'T230', 
            'R226', 'R222', 'Pol218', 'Plo214', 'B214', 
            'Pol214', 'Plo210', 'B210', 'Pol210', 'Plo206'
            ]

half_lives = [
            4.468e9*525600, 24.10*1440, 6.7*60, 
            245500*525600, 75380*525600, 1600*525600, 
            3.8235*1440, 3.10, 26.8, 19.9, 164.3/(6e7), 
            22.3*525600, 5015*525600, 138376*1440
            ]

num_products = len(half_lives) + 1

#Define ODEs

def fun(t, N, half_life_arr = half_lives):

    global num_products
    
    dNdt = np.zeros(num_products)
    
    #Middle elements decay but also increases due to decay of previous product
    for i in range(num_products):

        #First element purely decays
        if i == 0: 
            dNdt[i] = -N[i]/half_life_arr[i]

        #Last element is stable, so it doesn't decay.
        elif i == num_products-1:
            dNdt[i] = N[i-1]/half_life_arr[i-1]

        #Middle elements decay but also receive decay of prior product. 
        else: 
            dNdt[i] = N[i-1]/half_life_arr[i-1] - N[i]/half_life_arr[i]
    
    return dNdt*np.log(2)


#Start with pure U-238

N_0 = np.zeros(num_products)
N_0[0] = 1


#Set start and end time. I chose one half-life of U238
t0 = 0
t1 = half_lives[0]

#Calculate the amount of each product after half-life of U238 has elapsed, N(t1).
decay_products = integrate.solve_ivp(fun, [t0,t1], N_0, method = 'Radau')
N_t1 = decay_products.y

print('''PART A:
------    ''')

print('Began with 1 sample of U238. After one half life of U238, the amount of each product is given below:\n')

total_sum = 0
for i in range(num_products):
    print('{0}: {1}'.format(products[i], N_t1[i, -1]))
    total_sum += N_t1[i, -1]
print('\nTotal sum of all products (expected 1): {0}'.format(total_sum))


#------------#
#   PART B   #
#------------#

def fun2(t, N, half_life):

    #A simpler version of fun() defined above. Hardwired knowing that we only have a single decay product. 
    dNdt = np.zeros(2)
    dNdt[0] = -N[0]/half_life
    dNdt[1] = N[0]/half_life

    return dNdt*np.log(2)

#i) U238 --> Pb206
half_life = half_lives[0]
N_0 = np.array([1.0, 0.0]) #Initial values. Assuming it decays instantly into Pb. 
t0 = 0
t1 = half_life

def temp(t, N, half_life = half_life):
    return fun2(t, N, half_life)

#Get amount of decayed Pb over one half-life of U238
decay_products = integrate.solve_ivp(temp, [t0, t1], N_0, method = 'Radau')
N_t = decay_products.y


#Plot
plt.plot(decay_products.t, N_t[1]/N_t[0])
plt.xlabel('Time, minutes')
plt.ylabel('Pb206/U238')
plt.savefig('./figure1.pdf')
plt.show()

#ii) U234 --> T230
half_life = half_lives[4]
N_0 = np.array([1.0, 0.0]) #Initial values.
t0 = 0
t1 = half_life

def temp(t, N, half_life = half_life):
    return fun2(t, N, half_life)

#Get amount of decayed T230 over one half-life of U234
decay_products = integrate.solve_ivp(temp, [t0, t1], N_0, method = 'Radau')
N_t = decay_products.y

#Plot ratio
plt.plot(decay_products.t, N_t[1]/N_t[0])
plt.xlabel('Time, minutes')
plt.ylabel('T230/U234')
plt.savefig('./figure2.pdf')
plt.show()