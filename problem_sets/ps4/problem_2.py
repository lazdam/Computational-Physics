import matplotlib.pyplot as plt
import numpy as np
import camb
import time


#Import data 
planck = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
npts = len(spec)
errs=0.5*(planck[:,2]+planck[:,3]);


#Starting parameters: 
pars=np.asarray([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])
npars = len(pars)

def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:npts + 2]

def get_deriv(pars):

    derivs = np.empty([npts, npars])

    for j in range(npars):

        h_arr = np.zeros(npars)
        h_arr[j] = 0.01*float(pars[j])
        
        #Compute double sided derivative
        derivs[:, j] = (get_spectrum(pars+h_arr) - get_spectrum(pars - h_arr))/(2*h_arr[j])

    return derivs


def fit_newton(pars, chi_tol = 0.01, max_iterations = 10):

    #Get first model
    model = get_spectrum(pars)
    derivs = get_deriv(pars)


    r = spec - model

    lhs = derivs.T@derivs
    rhs = derivs.T@r

    dm = np.linalg.inv(lhs)@rhs
    for k in [0,1,2,5]:
        pars[k]+=dm[k]
    chisq_cur=np.sum((r/errs)**2)

    m = 0


    while True: 

        model = get_spectrum(pars)
        derivs = get_deriv(pars)

        r  = spec - model

        lhs = derivs.T@derivs

        cov = np.linalg.inv(lhs)

        errors = np.sqrt(np.diag(cov))

        rhs = derivs.T@r

        dm = cov@rhs
        chisq_new = np.sum((r/errs)**2)

        print('Current chisq:', chisq_cur)
        print('New chisq:', chisq_new)

        if np.abs(chisq_cur - chisq_new) < chi_tol:
            print('Achieved best fit parameters.')
            break 

        elif m == max_iterations:
            print('Exceeded maximum number of iterations.')
            break 

        else: 
            for k in [0,1,2,5]:
                pars[k]+=dm[k]

            chisq_cur = chisq_new 
            m+=1



    return pars, errors



best_fit, errors = fit_newton(pars)
print('''
Best fit parameters: {0}
Errors: {1}
'''.format(pars, errors))

