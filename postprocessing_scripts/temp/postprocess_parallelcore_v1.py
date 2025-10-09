from FPCSLpy.case import LargeCase
from mpi4py import MPI
import numpy as np
import os
import csv
import argparse
import re, pathlib


def grep_ctr(st, ctr_file="incompressible_tml.ctr"):
    """ To grep all the required data from the CTR file
    
    Args:
        st (string) : The variable whose data is to be grep-ed
    
    Return:
        The variable's value
    """
    
    text = pathlib.Path(ctr_file).read_text()
    pat = re.compile(rf"\b{re.escape(st)}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    m = pat.search(text)
    n   = float(m.group(1)) if m else None

    return n


def plot_thickness(time_steps, delta, x_label, y_label, file_name):
    """ Plot growth rate vs normalized time
    
    Args:
        time_steps (list)   : Normalized time
        delta (list)        : Normalized momentum, phi or mixing layer thickness
        x_label (string)    : X label
        y_label (string)    : Y label
        file_name (string)  : File Name
    
    Return:
        .png plots 
    
    Examples:
        plot_thickness(T, delta, 'Normalized Time', 'Momentum Thickness', 'momentum_thickness.png')
    """
        
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, delta_theta_g, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def momentum_thickness(U, u_bar, delta_u, alpha, dy, delta):
    """ Compute momentum thickness at each timestep per \\cite{lee2025effects}.
    
    Args:
        U       (float) : Gas(or phase1) free stream velocity
        u_bar   (float) : Reynolds averaged 
        delta_u (float) : Difference in free stream velocities
        alpha   (float) : Volume fraction
        dy      (float) : Cell(?) size along y direction

    Return:
        delta_theta (float) : Momentum thickness for a given time-step
    
    """

    integrand                = ((alpha) * (U - u_bar) * (u_bar))/(delta_u * delta_u)
    delta_theta_g            = (np.trapezoid(integrand, dx=dy))
    delta_theta_g_normalized = delta_theta_g / delta

    return delta_theta_g_normalized


def phi_thickness(alpha, nx_g_half, dy):
    """ Computing vol fraction thickness for insight into entrainement of one
        phase into another.
    
    Args:
        alpha     (float)   : Volume fraction
        nx_g_half (int)     : Half the grid size of the given simulation
    
    Return:
        delta_phi_g (float) : Volume fraction based thickness
    """

    integrand_phi           = alpha
    integrand_phi_subset    = integrand_phi[:nx_g_half]
    delta_phi_g             = np.trapezoid(integrand_phi_subset, dx=dy)
    
    return delta_phi_g


def mixinglayer_thickness(alpha, dy):
    """ Computing the mixing layer thickness
    
    Args:
        alpha (float) : Volume fraction
        dy    (float) : Cell(?) size along y direction

    Return:
        delta_mixing (float) : Mixing layer thickness
    """
    
    integrand_mixing = alpha * (1 - alpha)
    delta_mixing     = np.trapezoid(integrand_mixing, dx=dy)
    
    return delta_mixing
    

def case_update(ctr_file):
    """ Getting case files data
    
    Args:
        ctr_file (string) : CTR file path
    
    Return:
        nx_g (int)  : Grid nodes in x direction
        ny_g (int)  : Grid nodes in y direction
        nz_g (int)  : Grid nodes in z direction
        dx   (float): Discretization size in x direction
        dy   (float): Discretization size in y direction
        dz   (float): Discretization size in z direction
        case (Case) : FPCSL case object(?)
    
    """

    case = LargeCase(path='./.')
    #if case.rank == 0:
        #print("Starting")

    #Grid
    nx_g = int(grep_ctr("nx"))
    ny_g = int(grep_ctr("ny"))
    nz_g = int(grep_ctr("nz"))
    #Domain
    xmax = grep_ctr("xmax")
    a    = xmax / np.pi
    ymax = grep_ctr("ymax")
    b    = ymax / np.pi
    zmax = grep_ctr("zmax")
    c    = zmax / np.pi
    #Parallel
    nxsd = int(grep_ctr("nxsd"))
    nysd = int(grep_ctr("nysd"))
    nzsd = int(grep_ctr("nzsd"))

    #Update default parameters
    to_update_parameters = case.parameters
    to_update_parameters['grid']['x_max']                                       = a * np.pi
    to_update_parameters['grid']['y_max']                                       = b * np.pi
    #to_update_parameters['grid']['z_max']                                       = 5 * ((2 * np.pi) / nx_g)
    to_update_parameters['grid']['z_max']                                       = c * np.pi
    to_update_parameters['grid']['nx']                                          = nx_g
    to_update_parameters['grid']['ny']                                          = ny_g
    to_update_parameters['grid']['nz']                                          = nz_g
    to_update_parameters['simulation_parameters']['parallel']['nxsd']           = nxsd
    to_update_parameters['simulation_parameters']['parallel']['nysd']           = nysd
    to_update_parameters['simulation_parameters']['parallel']['nzsd']           = nzsd
    to_update_parameters['simulation_parameters']['solvers']['incompressible']  = True

    #Update and check parameters
    case.update_parameters(to_update_parameters)

    #dx, dy, dz = case.grid['dx'], case.grid['dy'], case.grid['dz']
    dx = (case.parameters['grid']['x_max'] - case.parameters['grid']['x_min']) / case.parameters['grid']['nx']
    dy = (case.parameters['grid']['y_max'] - case.parameters['grid']['y_min']) / case.parameters['grid']['ny']
    dz = (case.parameters['grid']['z_max'] - case.parameters['grid']['z_min']) / case.parameters['grid']['nz']

    return  nx_g, ny_g, nz_g, dx, dy, dz, case


def split_timestep_over_cores(delta, U_g, dt, delta_u,
                
                              nx_g, ny_g, nz_g,
                              nx_g_half,
                              dx, dy, dz,
                              case,
                              
                              time_step):
    """
    
    Args:
    
    Return:
    
    """

    #Change the arguement to this fn to something passed from main
    case.distribute_block_list_axis_stack(1)
    filtered_block_list = case.filtered_rank_block_list

    #Hardcoding with nxsd since I see that it has z-x arrangement, change it later
    nxsd = int(grep_ctr("nxsd"))
    nysd = int(grep_ctr("nysd"))
    nzsd = int(grep_ctr("nzsd"))

    #Maybe change this from being a hardcoded size?
    a = int(nx_g / nxsd)
    b = int(ny_g / nysd)
    c = int(nz_g / nzsd)
    u_block   = np.empty  ((a, b, c, nzsd, nxsd))
    alpha_block = np.empty((a, b, c, nzsd, nxsd))
    for i in range(nxsd):
        for k in range(nzsd): 
            nxr, nyr, nzr = case.get_nxrnyrnzr_from_nr(filtered_block_list[i * nzsd + k])
            block         = case.read_block(time_step, nxr, nyr, nzr, to_read=['u', 'phi_2'], to_interpolate=True)
            u     = block['u']
            phi_2 = block['phi_2']
            
            u_block[..., k, i]   = u    
            alpha_block[..., k, i] = 1 - phi_2    
            
            del(block)
    
    #Averaging
    #Could you once again check why you are averaging out the alpha?
    u_bar = np.mean(u_block, axis=(0, 2, 3, 4)) 
    alpha = np.mean(alpha_block, axis=(0, 2, 3, 4)) 
    
    local_mt    = momentum_thickness(U_g, u_bar, delta_u, alpha, dy, delta)
    local_pt    = phi_thickness(alpha, nx_g_half, dy)
    local_mixt  = mixinglayer_thickness(alpha, dy)

    #if(case.rank == 0):
    global_mt   = case.comm.reduce(local_mt, op=MPI.SUM, root=0)
    global_pt   = case.comm.reduce(local_pt, op=MPI.SUM, root=0)
    global_mixt = case.comm.reduce(local_mixt, op=MPI.SUM, root=0)

    if(case.rank == 0):
        print(global_mt)
        print(global_pt)
        print(global_mixt)
    
    quit()
    
    return  momentum_thickness,  \
            phi_thickness,       \
            mixinglayer_thickness


def main():
    """
    
    Args:
    
    Return:

    """
    
    #Variables
    delta_theta_g   = []
    delta_phi_g     = []
    delta_mixing    = []
    delta           = (2. * np.pi) / 100.
    U_g             = 3.1830988618379066
    U_l             = 0.
    delta_u         = U_g - U_l

    dt              = 0.001

    cores           = 1

    start_ts        = 0
    step_ts         = 2000
    end_ts          = 2000

    ctr_file        = "incompressible_tml.ctr"

    (nx_g, ny_g, nz_g,
     dx, dy, dz,       
     case)             = case_update(ctr_file)
    nx_g_half          = int(nx_g / 2)

    #For time-steps_{i}
    time_steps = [i for i in range(start_ts, end_ts, step_ts)]
    
    for time_step in time_steps:

        (momentum_thickness,  
        phi_thickness,       
        mixinglayer_thickness) = split_timestep_over_cores(delta, U_g, dt, delta_u,
                    
                                                           nx_g, ny_g, nz_g,
                                                           nx_g_half,
                                                           dx, dy, dz,
                                                           case,
                                                           
                                                           time_step)

    return momentum_thickness, phi_thickness, mixinglayer_thickness


if __name__ == "__main__":
    (momentum_thickness,  
    phi_thickness,       
    mixinglayer_thickness) = main()

    print(mixinglayer_thickness)
