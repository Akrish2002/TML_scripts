import numpy as np
from scipy.optimize import fsolve

def initialize_array(y_min, y_max, ny):
    """
    Create a 1D grid using linspace and calculate the grid spacing dy.
    """
    y = np.linspace(y_min, y_max, ny)
    dy = (y_max - y_min) / (ny - 1)  # ensure dy is consistent with linspace
    return y, dy

def u_profile(y, y_ref, Ug, Ur, delta, c):
    """
    Construct the velocity profile using a hyperbolic tangent.
    
    Parameters:
      y    : 1D grid array
      y_ref: reference y-value for the profile
      Ug   : free-stream velocity
      Ur   : lower region velocity
      delta: scaling parameter (e.g., a thickness)
      c    : parameter controlling the profile steepness
    """
    idx1 = np.where(y - y_ref >= 0)
    idx2 = np.where(y - y_ref < 0)

    u = np.empty_like(y)
    u[idx1] = (Ug - Ur) * np.tanh((y[idx1] - y_ref) / (c * delta)) + Ur
    u[idx2] = Ur * np.tanh((y[idx2] - y_ref) / (c * delta)) + Ur
    return u  

def momentum_thickness(Ug, u, dy, alpha=1):
    """
    Compute the momentum thickness via numerical integration.
    
    Parameters:
      Ug   : free-stream velocity
      u    : velocity profile array
      dy   : grid spacing
      alpha: Vol frac
    
    Returns:
      The computed momentum thickness.
    """
    #print(alpha.shape, alpha.dtype)
    #print(u.shape, u.dtype)
    integrand = alpha * (Ug - u) * u / (Ug*Ug)
    delta_theta = np.trapz(integrand, dx=dy)
    return delta_theta

def get_optimal_c(y, dy, y_ref, Ug, Ur, delta, initial_guess=3.26):
    """
    Solve for the optimal c such that the normalized momentum thickness equals 1.
    
    Parameters
    ----------
      y             : 1D grid array
      dy            : grid spacing
      y_ref         : reference y-value for the profile
      Ug            : free-stream velocity
      Ur            : velocity in the lower region
      delta         : scaling parameter (e.g., layer thickness)
      initial_guess : starting guess for c (default 3.26)
    
    Returns
    -------
      The optimal value of c.
    """
    y = np.mean(y, axis=(0,2))
    def f(c):
        c = float(c)
        u = u_profile(y, y_ref, Ug, Ur, delta, c)
        theta = momentum_thickness(Ug, u, dy)
        return theta / delta - 1

    return fsolve(f, initial_guess)[0]

#if __name__ == '__main__':
    # Example usage with input parameters.
    # In practice, these values would be passed from initialize_tml.py.
    
    # Define input parameters
    #y_min = 0
    #y_max = 2 * np.pi
    #ny = 1024
    #y_ref = np.pi
    #delta = 2 * np.pi / 100.
    #
    ## Define velocity parameters (these should match your simulation)
    #Ug = 3.18309  # typically U1, the free-stream velocity
    #Ur = 0.0      # for example, U2/U1; adjust as needed
    #
    ## Create the grid
    #y, dy = initialize_array(y_min, y_max, ny)
    #
    ## Compute and print the optimal c
    #c_optimal = get_optimal_c(y, dy, y_ref, Ug, Ur, delta)
    #print("Optimal c:", c_optimal)

