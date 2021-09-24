import numpy as np

def integrate_adaptive(fun, a, b, tol, extra = None):
    '''
    Adaptive integrator for a function that does not repeat function evaluations on previously computted values. 

    Parameters:
    -----------
    fun: function, callable
        Function you'd like to integrate
    a: float
        Lower bount of integration
    b: float
        Upper bound of integration
    tol: float
        Error tolerance between successive Simpson's areas.  

    extra: bool
        Is none only for first function call. For recursive calls, extra = ndarray containing saved data points. 

    Returns: Integral from a to b. 

    '''
    
    if type(extra) == type(None): 

        #First run through, calculate initial set of points and areas. 
        
        x = np.linspace(a,b,5)
        y = fun(x)
        
        #Code from class
        dx=(b-a)/(len(x)-1)
        area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
        area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
        err=np.abs(area1-area2)
        
        saved_data = np.vstack((x,y)) #Stores previously computted x and y values. 
                                      #If error isn't below tolerance level, this data will be carried over to the next function call. 
        
        global new_counter #Counts number of function calls for improved adaptive integrator
        global old_counter #Counts number of function calls for old adaptive integrator seen in class 

        new_counter+=len(x) #Increase the number of new function calls
        old_counter+=len(x) #Increase the number of old function calls (recalling that each time, it always computted fun(x) ==> len(x) function calls)

        
        if err<tol: 
            return area2

        else: 
            midpoint = (a + b)/2
            left = integrate_adaptive(fun, a, midpoint, tol/2, extra = saved_data) #saved data will be used in next step
            right = integrate_adaptive(fun, midpoint, b, tol/2, extra = saved_data) #same here
            
            return left + right
    
    #For subsequent recursive calls
    else: 
        
        #Extract saved x and y values from previous function calls
        saved_data = extra
        saved_x = saved_data[0] 
        saved_y = saved_data[1]

        
        #Generate x values as usual. Need to fill up y array with either new or old values. 
        x = np.linspace(a,b,5)
        y = np.empty(len(x))

        
        for i, x_i in enumerate(x):
            
            if x_i in saved_x: #Checks to see if x_i has already been generated
                
                index = np.where(saved_x == x_i)[0][0] #Finds location of x_i in saved x values
                y[i] = saved_y[index] #Retrieve already calculated y value corresponding to x_i
            
            else: #If not, calculate new y value
                
                saved_x = np.append(saved_x, x_i) #Save the x value for future reference
                y[i] = fun(x_i) #Calculate the new y value
                saved_y = np.append(saved_y, y[i]) #Save the y value for future reference
               
                new_counter+=1 #Count number of times we evaluate a new function
            
        
        old_counter+=len(x) #Old adaptive integrator would've computted f(x) (equivalent to an increase of len(x) function calls)


        saved_data = np.vstack((saved_x, saved_y)) #stack new saved x and y values for next loop, if needed
        
        #Now that we've got our x and y values, we can compute areas and error as before
        dx=(b-a)/(len(x)-1)
        area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
        area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
        err=np.abs(area1-area2)
        
        if err < tol: 
            return area2
        
        
        else:
            midpoint = (a + b)/2
            left = integrate_adaptive(fun, a, midpoint, tol/2, extra = saved_data)
            right = integrate_adaptive(fun, midpoint, b, tol/2, extra = saved_data)

            return left + right


#Few typical examples

#f(x) = exp(x)
def fun(x):
    return np.exp(x)

a = 0
b = 1
new_counter = 0
old_counter = 0

integral = integrate_adaptive(fun, a, b, 1e-7)

print('''The integral of f(x) = exp(x) from {0} to {1} is {2}. We performed {3} function calls as compared to the {4} function calls performed in class (saved {5} function calls!).\n'''.format(a,b,integral,new_counter, old_counter, old_counter - new_counter))


#f(x) = Lorentzian 
def fun(x):
    return 1/(1 + x**2)

a = -100
b = 100
new_counter = 0
old_counter = 0

integral = integrate_adaptive(fun, a, b, 1e-7)

print('''The integral of f(x) = 1/(1 + x^2) from {0} to {1} is {2}. We performed {3} function calls as compared to the {4} function calls performed in class (saved {5} function calls!).\n'''.format(a,b,integral,new_counter, old_counter, old_counter - new_counter))
