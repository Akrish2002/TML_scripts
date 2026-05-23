import numpy as np
from scipy.optimize import fsolve

def initialize_array(y_min, y_max, ny):
    """
    Create a 1D grid using linspace and calculate the grid spacing dy.
    """
    y = np.linspace(y_min, y_max, ny)
    dy = (y_max - y_min) / (ny - 1)  # ensure dy is consistent with linspace
    return y, dy

def u_profile(y, y_ref, delta_u, delta, c):
    """
    Construct the velocity profile using a hyperbolic tangent.
    
    Parameters:
      y    : 1D grid array
      y_ref: reference y-value for the profile
      Ug   : free-stream velocity
      delta: scaling parameter (e.g., a thickness)
      c    : parameter controlling the profile steepness
    """
    u = np.empty_like(y)
    u = delta_u * 0.5 * np.tanh((y - y_ref) / (c * delta)) 
    return u  

def momentum_thickness(U1, u, dy, delta_u):
    """
    Compute the momentum thickness via numerical integration.
    
    Parameters:
      Ug   : free-stream velocity
      u    : velocity profile array
      dy   : grid spacing
      alpha: asdf?
    
    Returns:
      The computed momentum thickness.
    """
    integrand = (U1 * U1 - u * u) / (delta_u * delta_u)
    delta_theta = np.trapz(integrand, dx=dy)
    return delta_theta

def get_optimal_c(y, dy, y_ref, U1, delta_u, delta, initial_guess=3.26):
    """
    Solve for the optimal c such that the normalized momentum thickness equals 1.
    
    Parameters
    ----------
      y             : 1D grid array
      dy            : grid spacing
      y_ref         : reference y-value for the profile
      U1            : free-stream velocity
      delta         : scaling parameter (e.g., layer thickness)
      initial_guess : starting guess for c (default 3.26)
    
    Returns
    -------
      The optimal value of c.
    """
    y = np.mean(y, axis=(0,2))
    def f(c):
        c = float(c)
        u = u_profile(y, y_ref, delta_u, delta, c)
        theta = momentum_thickness(U1, u, dy, delta_u)
        return theta / delta - 1

    return fsolve(f, initial_guess)[0]

