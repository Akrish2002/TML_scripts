from concurrent.futures import ProcessPoolExecutor as ex
from FPCSLpy.case import Case
import numpy as np
import os
import csv
import argparse


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


def momentum_thickness(U, u_bar, delta_u, alpha, dy):
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

    integrand     = ((alpha) * (U_g - u_bar) * (u_bar))/(delta_u * delta_u)
    delta_theta_g = (np.trapz(integrand, dx=dy))
    
    return delta_theta_g


def phi_thickness(alpha, nx_g_half):
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


def mixinglayer_thickness(alpha, dy)
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
    

def case_update():
    case = Case(path='./.')

    args = parse_args()
    nx_g, ny_g, nz_g = args.nx_g, args.ny_g, args.nz_g

    #Update default parameters
    to_update_parameters = case.parameters
    to_update_parameters['grid']['x_max']                                       = 2. * np.pi
    to_update_parameters['grid']['y_max']                                       = 2. * np.pi
    to_update_parameters['grid']['z_max']                                       = 2. * np.pi
    to_update_parameters['grid']['nx']                                          = nx_g
    to_update_parameters['grid']['ny']                                          = ny_g
    to_update_parameters['grid']['nz']                                          = nz_g
    to_update_parameters['simulation_parameters']['parallel']['nxsd']           = 32
    to_update_parameters['simulation_parameters']['parallel']['nysd']           = 32
    to_update_parameters['simulation_parameters']['parallel']['nzsd']           = 32
    to_update_parameters['simulation_parameters']['solvers']['incompressible']  = True

    dx, dy, dz = case.grid['dx'], case.grid['dy'], case.grid['dz']

    #Update and check parameters
    case.update_parameters(to_update_parameters)

    return  nx_g, ny_g, nz_g \
            dx, dy, dz


def main():
    
if __name__ == "__main__":

    #Variables
    delta_theta_g   = []
    delta_phi_g     = []
    delta_mixing    = []

    delta           = (2. * np.pi) / 100.
    U_g             = 3.1830988618379066
    dt              = 0.00025

    main()
