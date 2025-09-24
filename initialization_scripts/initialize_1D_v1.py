import numpy as np
from scipy.optimize import fsolve

def initialize_array(y_min, y_max, ny):

    dy = (y_max-y_min)/ny
    return np.arange(y_min, y_max + dy, dy), dy

def u_profile(y, y_ref, Ug, Ur, delta, c):

    idx1 = np.where(y - y_ref >= 0)
    idx2 = np.where(y - y_ref < 0)

    u = np.empty_like(y)
    
    y_calc = ((y[idx1]-y_ref)/(c*delta))
    u[idx1] = (Ug - Ur)*np.tanh((y[idx1]-y_ref)/(c*delta)) + Ur
    u[idx2] = Ur*np.tanh((y[idx2]-y_ref)/(c*delta)) + Ur
    return u  

def momentum_thickness(Ug, u, dy, alpha = 1):
    
    integrand = alpha * (Ug - u)*(u)/(Ug*Ug)
    delta_theta_g = np.trapz(integrand, dx = dy)
    return delta_theta_g

def parameters(y, dy, y_ref, U1, Ur, delta, c):
    
    y = np.mean(y, axis=(0,2)) 
    u = u_profile(y, y_ref, U1, Ur, delta, c)
    theta = momentum_thickness(U1, u, dy)
    return theta

def solve_f(y, dy, y_ref, Ug, Ur, delta, initial_guess=3.26):
    
    y = np.mean(y, axis=(0,2)) 
    def f(c):
        u = u_profile(y, y_ref, Ug, Ur, delta, c)
        theta = momentum_thickness(Ug, u, dy)
        return theta / delta - 1
  
    c = fsolve(f, 3.26)[0]
    return c

if __name__ == "__main__":

    #Parameters
    y_min = 0
    y_max = 2*np.pi
    y_ref = np.pi
    ny = 1024
    rho1, rho2 = 1., 1.
    mu1 , mu2  = 1.e-3, 5.e-2
    U1  , U2   = 1., 0.
    V1  , V2   = 0., 0.
    delta = 2.*np.pi/100.
    Re = 200.
    We = 20.
    c = 3.2593927029

#y, dy = initialize_array(y_min, y_max, ny)

