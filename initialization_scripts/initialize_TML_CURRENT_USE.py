import numpy as np
import matplotlib.pyplot as plt
import initialize_1D_v2 as initialize_1D
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
                                        description='Initialize grid parameters and solve for optimal c'
                                    )

    parser.add_argument(
                            '--nx_g', type=int, required=True,
                            help='Number of grid nodes in the x-direction')

    parser.add_argument(
                            '--ny_g', type=int, required=True,
                            help='Number of grid nodes in the y-direction'
                       )

    parser.add_argument(
                            '--nz_g', type=int, required=True,
                            help='Number of grid nodes in the z-direction'
                       )

    parser.add_argument(
                            '--nxsd', type=int, required=True,
                       )

    parser.add_argument(
                            '--nysd', type=int, required=True,
                       )

    parser.add_argument(
                            '--nzsd', type=int, required=True,
                       )


    return parser.parse_args()

def add_random_noise(f):
    noise = 0.05*f*np.random.normal(0, scale=1, size=f.shape)
    return f+noise

def add_random_noise_based_ref(f, g):
    noise = 0.05*f*np.random.normal(0, scale=1, size=f.shape)
    return g+noise

def params(Re, mu, delta, rho, We):
    u1 = (Re * mu)/delta #Re = (U1 * delta)/mu
    sigma = (rho * u1 * u1 * delta)/We #We = (rho * u1 * u1 * delta)/sigma
    print("--U1=",u1, "--Sigma=", sigma)
    return u1, sigma

if __name__ == '__main__':
    from FPCSLpy.case import Case
    case = Case(path='./.')

    args = parse_args()
    nx_g, ny_g, nz_g = args.nx_g, args.ny_g, args.nz_g
    nxsd, nysd, nzsd = args.nxsd, args.nysd, args.nzsd

    # Update default parameters
    to_update_parameters = case.parameters
    to_update_parameters['grid']['x_max'] = 2.*np.pi
    to_update_parameters['grid']['y_max'] = 2.*np.pi
    to_update_parameters['grid']['z_max'] = 2.*np.pi
    to_update_parameters['grid']['nx'] = nx_g
    to_update_parameters['grid']['ny'] = ny_g
    to_update_parameters['grid']['nz'] = nz_g
    to_update_parameters['simulation_parameters']['parallel']['nxsd'] = nxsd
    to_update_parameters['simulation_parameters']['parallel']['nysd'] = nysd
    to_update_parameters['simulation_parameters']['parallel']['nzsd'] = nzsd
    to_update_parameters['simulation_parameters']['solvers']['incompressible'] = True

    # Update and check parameters   
    case.update_parameters(to_update_parameters)

    # Parameters
    rho1, rho2 = 1., 1.
    mu1 , mu2  = 1.e-3, 5.e-2
    U1  , U2   = 1., 0.
    V1  , V2   = 0., 0.
    yloc = np.pi
    epsilon = 0.51*case.grid['dx']
    delta = 2.*np.pi/100.
    Re = 200.
    We = 20.

    U1, sigma = params(Re, mu1, delta, rho1, We)
    Ur = U2/U1

    # Create custom initial conditions
    time_step = 0
    print("--", end="")
    case.create_default_fields(time_step)
    custom = case.data[f'{time_step}']

    zeros = np.zeros_like(case.grid['xcs'])
    ones  = np.ones_like(case.grid['xcs'])

    phi1 = 0.5*(1.+np.tanh((case.grid['ycs']-yloc)/(2.*epsilon)))
    phi2 = 1. - phi1
    rho  = phi1*rho1 + (1. - phi1)*rho2
    mu   = phi1*mu1  + (1. - phi1)*mu2
    #mix1 = 0.5*(1.+np.tanh((case.grid['ycs']-yloc)/(2.*delta)))
    idx1  = np.where(case.grid['ycs']-yloc>=0.)
    idx2  = np.where(case.grid['ycs']-yloc<0.)

    c = initialize_1D.get_optimal_c(case.grid['ycs'], (2*np.pi/nx_g), yloc, U1, Ur, delta)
    print("--Computed c val: ", c)

    u1 = (U1-Ur)*np.tanh((case.grid['ycs'][idx1]-yloc)/(c*delta)) + Ur
    u2 = Ur*np.tanh((case.grid['ycs'][idx2]-yloc)/(c*delta)) + Ur
    u  = np.zeros_like(case.grid['ycs'])
    u[idx1] = u1
    u[idx2] = u2
    v    = phi1*V1 + (1. - phi1)*V2

    u1_t = np.mean(u, axis=(0,2))
    alpha = np.mean(1-(case.data[f'{time_step}']['phi_2']), axis=(0,2))
    print("--Normalized momentum thickness: ", initialize_1D.momentum_thickness(U1, u1_t, ((2*np.pi)/nx_g), alpha)/delta)

    unoise = add_random_noise(u)
    vnoise = add_random_noise_based_ref(u, v)
    wnoise = add_random_noise_based_ref(u, v)

    print("--U min & max val: ", unoise.min(), unoise.max())
    print("--V min & max val: ", vnoise.min(), vnoise.max())
    print("--W min & max val: ", wnoise.min(), wnoise.max())

    idx = int(case.grid['nx']/2.)
    fig, ax1 = plt.subplots(figsize=(3,3))
    ax1.plot(phi1[idx,:,2], case.grid['ycs'][idx,:,2], 'k')
    ax1.set(
        xlabel = r'$\phi_1$ [-]',
        ylabel = r'$y$ [m]',
    )
    ax2 = ax1.twiny()
    ax2.plot(u[idx,:,2], case.grid['ycs'][idx,:,2], 'r--')
    ax2.plot(unoise[idx,:,2], case.grid['ycs'][idx,:,2], 'g--')
    ax2.set(
        xlabel = r'$u$ [m/s]',
    )
    fig.tight_layout()
    fig.savefig(f'./profiles_u.png', dpi=300)
    plt.close(fig)

    idx = int(case.grid['nx']/2.)
    fig, ax1 = plt.subplots(figsize=(3,3))
    ax1.plot(phi1[idx,:,2], case.grid['ycs'][idx,:,2], 'k')
    ax1.set(
        xlabel = r'$\phi_1$ [-]',
        ylabel = r'$y$ [m]',
    )
    ax2 = ax1.twiny()
    ax2.plot(v[idx,:,2], case.grid['ycs'][idx,:,2], 'r--')
    ax2.plot(vnoise[idx,:,2], case.grid['ycs'][idx,:,2], 'g--')
    ax2.set(
        xlabel = r'$v$ [m/s]',
    )
    fig.tight_layout()
    fig.savefig(f'./profiles_v.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4,3))
    cs = ax.pcolormesh(case.grid['xcs'][:,:,2], case.grid['ycs'][:,:,2], unoise[:,:,2], cmap='inferno')
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label(r'$u_1$ [m/s]')
    #ax.contour(xcs, ycs, phi1, colors='k', levels=[0.5])
    ax.set(
        xlabel = r'$x$ [m/s]',
        ylabel = r'$y$ [m/s]',
    )
    fig.tight_layout()
    fig.savefig(f'./noise.png', dpi=300)
    #plt.show()
    plt.close(fig)

    custom['u'] = unoise
    custom['v'] = vnoise
    custom['w'] = wnoise
    custom['phi_1'] = phi1
    custom['phi_2'] = phi2
    custom['rho'] = rho
    custom['p'] = ones
    custom['mu'] = mu
    print("--", end="")
    case.write_custom_fields(time_step, custom, to_interpolate=True)

