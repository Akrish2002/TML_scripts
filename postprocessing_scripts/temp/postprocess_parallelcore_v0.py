from concurrent.futures import ProcessPoolExecutor
from FPCSLpy.case import Case
import numpy as np
import os
import csv
import argparse
import re, pathlib


def parse_args():
    """ Parse arguments from command line
    
    Args:
    
    Return:
        argparse.Namespace: Parsed arguments with attributes:
            nx_g : Number of grid nodes in x
            ny_g : Number of grid nodes in y
            nz_g : Number of grid nodes in z

    Examples:

        $ python3 postprocess.py --nx_g 1024 --ny_g 1024 --nz_g 1024
    """

    parser = argparse.ArgumentParser(
                                        description='Grid'
                                    )

    parser.add_argument(
                            '--nx_g', type=int, required=True,
                            help='Number of grid nodes in the x-direction'
                       )

    parser.add_argument(
                            '--ny_g', type=int, required=True,
                            help='Number of grid nodes in the y-direction'
                       )

    parser.add_argument(
                            '--nz_g', type=int, required=True,
                            help='Number of grid nodes in the z-direction'
                       )
    return parser.parse_args()


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
    delta_theta_g            = (np.trapz(integrand, dx=dy))
    delta_theta_g_normalized = delta_theat_g / delta

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
    delta_phi_g             = np.trapz(integrand_phi_subset, dx=dy)
    
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
    delta_mixing     = np.trapz(integrand_mixing, dx=dy)
    
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

    case        = Case(path='./.')

    #args = parse_args()
    #nx_g, ny_g, nz_g = args.nx_g, args.ny_g, args.nz_g

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
    to_update_parameters['grid']['z_max']                                       = 5 * ((2 * np.pi) / nx_g)
    to_update_parameters['grid']['nx']                                          = nx_g
    to_update_parameters['grid']['ny']                                          = ny_g
    to_update_parameters['grid']['nz']                                          = nz_g
    to_update_parameters['simulation_parameters']['parallel']['nxsd']           = nxsd
    to_update_parameters['simulation_parameters']['parallel']['nysd']           = nysd
    to_update_parameters['simulation_parameters']['parallel']['nzsd']           = nzsd
    to_update_parameters['simulation_parameters']['solvers']['incompressible']  = True

    #Update and check parameters
    case.update_parameters(to_update_parameters)

    dx, dy, dz = case.grid['dx'], case.grid['dy'], case.grid['dz']

    return  nx_g, ny_g, nz_g,   \
            dx, dy, dz,         \
            case


def call_cores( delta, U_g, dt, delta_u,
                
                nx_g, ny_g, nz_g,
                nx_g_half,
                dx, dy, dz,
                case,
                
                time_step):
    """
    
    Args:
    
    Return:
    
    """

    case.read_time_steps([time_step], to_interpolate=True)
    u, v, w  = case.data[f'{time_step}']['u'], \
               case.data[f'{time_step}']['v'], \
               case.data[f'{time_step}']['w']

    u_bar = np.mean(u, axis=(0, 2))                                             #Avg along x and z since it is periodic
    alpha = np.mean(1 - (case.data[f"{time_step}"]["phi_2"]), axis=(0, 2))      #1 - phi_2
    
    return  momentum_thickness(U_g, u_bar, delta_u, alpha, dy, delta), \
            phi_thickness(alpha, nx_g_half, dy), \
            mixinglayer_thickness(alpha, dy)


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

    dt              = 0.00025

    cores           = 15

    start_ts        = 0
    step_ts         = 800
    end_ts          = 31200

    ctr_file        = "incompressible_tml.ctr"

    (nx_g, ny_g, nz_g,
     dx, dy, dz,       
     case)             = case_update(ctr_file)
    nx_g_half          = int(nx_g / 2)

    #For time-steps_{i}
    time_steps = [i for i in range(start_ts, end_ts, step_ts)]

    with ProcessPoolExecutor(max_workers=cores) as ex: 
         futures  = [ex.submit(call_cores, 

                    delta, U_g, dt, delta_u,
                                    
                    nx_g, ny_g, nz_g,
                    nx_g_half,
                    dx, dy, dz,
                    case,
                                    
                    time_step) for time_step in time_steps]
 
    return futures


if __name__ == "__main__":
    futures = main()
    results = np.array([f.result() for f in futures])
    print(results)
