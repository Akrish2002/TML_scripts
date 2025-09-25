import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse

#def parse_args():
#    parser = argparse.ArgumentParser(
#                                        description='Grid'
#                                    )
#
#    parser.add_argument(
#                            '--nx_g', type=int, required=True,
#                            help='Number of grid nodes in the x-direction'
#                       )
#
#    parser.add_argument(
#                            '--ny_g', type=int, required=True,
#                            help='Number of grid nodes in the y-direction'
#                       )
#
#    return parser.parse_args()

nx_g = 1024
ny_g = 1024

def plot_thickness(time_steps, delta_theta_g, x_label, y_label, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, delta_theta_g, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

if __name__ == '__main__':
    from FPCSLpy.case import Case
    case = Case(path='./.')

    #args = parse_args()
    #nx_g, ny_g = args.nx_g, args.ny_g

    # Update default parameters
    to_update_parameters = case.parameters
    to_update_parameters['grid']['x_max'] = 2.*np.pi
    to_update_parameters['grid']['y_max'] = 2.*np.pi
    to_update_parameters['grid']['z_max'] = 5*(2.*np.pi/nx_g)
    to_update_parameters['grid']['nx'] = nx_g
    to_update_parameters['grid']['ny'] = ny_g
    to_update_parameters['grid']['nz'] = 5
    to_update_parameters['simulation_parameters']['parallel']['nxsd'] = 32
    to_update_parameters['simulation_parameters']['parallel']['nysd'] = 32
    to_update_parameters['simulation_parameters']['parallel']['nzsd'] = 1
    to_update_parameters['simulation_parameters']['solvers']['incompressible'] = True

    # Update and check parameters   
    case.update_parameters(to_update_parameters)
    
    delta_theta_g =[]
    delta_phi_g = []
    delta_mixing = []
    delta = (2.*np.pi)/100.
    U_g = 3.1830988618379066
    dt = 0.00025

    dx, dy, dz = case.grid['dx'], case.grid['dy'], case.grid['dz']

    for i in range(0, 31200, 800):
        #Time step
        time_step = i
        time_steps = [time_step]
        case.read_time_steps(time_steps, to_interpolate=True)

        u , v , w  = case.data[f'{time_step}']['uc'], \
                     case.data[f'{time_step}']['vc'], \
                     case.data[f'{time_step}']['wc']

        u1_t = np.mean(u, axis=(0,2))
        alpha = np.mean(1-(case.data[f'{time_step}']['phi_2']), axis=(0,2))

        #Momentum Thickness
        integrand_m = ((alpha) * (U_g - u1_t) * (u1_t))/(U_g * U_g)
        delta_theta_g.append(np.trapz(integrand_m, dx=dy))
        
        #Phi Thickness
        integrand_phi = (alpha)
        nx_g_half = 512
        integrand_phi_subset = integrand_phi[:nx_g_half]
        delta_phi_g.append(np.trapz(integrand_phi_subset, dx=dy))
        
        #mixing layer 
        integrand_mixing = alpha*(1-alpha)
        delta_mixing.append(np.trapz(integrand_mixing, dx=dy))

    time_steps = list(range(0, 31200, 800))
    t_normalized = list((t * dt * U_g)/delta for t in time_steps)

    delta_theta_g_normalized = np.array(delta_theta_g)/delta

    plot_thickness(t_normalized, delta_theta_g_normalized, 'Normalized Time', 'Momentum Thickness', 'momentum_thickness.png') 
    plot_thickness(t_normalized, delta_phi_g, 'Normalized Time', 'VolFrac Thickness', 'phi_thickness.png')
    plot_thickness(t_normalized, delta_mixing, 'Normalized Time', 'Mixing Layer Thickness', 'mixing_layer_thickness.png')

    with open(f"integrand_data_n{nx_g}.csv", "w", newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["Time", "Momentum", "Phi", "Mixing"])

        for t, m, phi, mixing in zip(time_steps, delta_theta_g_normalized, delta_phi_g, delta_mixing):
            writer.writerow([(t * dt * U_g)/delta, m, phi, mixing])
