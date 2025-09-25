from concurrent.futures import ProcesspoolExecutor as pool
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

def momentum_thickness(U, u, alpha):
    """ Compute momentum thickness at each timestep
    
    Args:
        U (float): Gas(or phase1) free stream velocity
        u (float):

    integrand_m = ((alpha) * (U_g - u1_t) * (u1_t))/(U_g * U_g)
    delta_theta_g.append(np.trapz(integrand_m, dx=dy))


def main():
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

    #Update and check parameters
    case.update_parameters(to_update_parameters)
    
    delta_theta_g   = []
    delta_phi_g     = []
    delta_mixing    = []

    delta           = (2. * np.pi) / 100.
    U_g             = 3.1830988618379066
    dt              = 0.00025
    dx, dy, dz      = case.grid['dx'], case.grid['dy'], case.grid['dz']

if __name__ == "__main__":
    main()
    print("Hello world") 
