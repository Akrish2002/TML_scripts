import numpy as np
import matplotlib.pyplot as plt
import initialize_1D_singlephase as initialize_1D
import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Initialize grid parameters and solve for optimal c')

    parser.add_argument('--nx_g', type=int, required=True, help='Number of grid nodes in the x-direction')
    parser.add_argument('--ny_g', type=int, required=True, help='Number of grid nodes in the x-direction')
    parser.add_argument('--nz_g', type=int, required=True, help='Number of grid nodes in the x-direction')

    parser.add_argument('--nxsd', type=int, required=True)
    parser.add_argument('--nysd', type=int, required=True)
    parser.add_argument('--nzsd', type=int, required=True)

    parser.add_argument('--amp',     type=float, required=True)

    parser.add_argument('--Re',      type=float, required=True)
    parser.add_argument('--delta_U', type=float, required=True)


    return parser.parse_args()

def generate_perturbations(Nz, Ny, Nx, dz, dy, dx, amp, delta, delta_U, seed=0):
    """
    Generate a 3D divergence-free velocity perturbation field (u', v', w')
    with a specified energy spectrum.
    """

    np.random.seed(seed)

    # Target perturbation amplitude.
    # Baltzer & Livescu prescribe 0.1 * Delta_U root-mean-square fluctuation
    # for each velocity component individually.
    amp_u = amp * delta_U
    amp_v = amp * delta_U
    amp_w = amp * delta_U

    kx = 2.0 * math.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * math.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2.0 * math.pi * np.fft.fftfreq(Nz, d=dz)

    # Initializing lambda_s
    lambda_ls = 28.0 * delta

    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")

    # Required for performing divergence-free projection below
    K2 = KX**2 + KY**2 + KZ**2
    k  = np.sqrt(K2)

    # k0 is a wavenumber, so it must be 2*pi divided by the target wavelength.
    lambda_peak = lambda_ls / 4.0
    k0          = 2.0 * math.pi / lambda_peak

    # You should define these from the box lengths if you want to keep k_eta.
    Lx = Nx * dx
    Ly = Ny * dy
    Lz = Nz * dz

    E_k = np.zeros_like(k)

    # This avoids adding any energy to the zero-mode
    mask = k > 0
    kk   = k[mask]
    E_k[mask] = (kk / k0)**4 * np.exp(-2.0 * (kk / k0)**2)

    theta        = 2.0 * np.pi * np.random.rand(3, Nz, Ny, Nx)
    random_phase = np.exp(1j * theta)
    A            = np.sqrt(E_k + 1e-30)
    U_hat        = random_phase * A[None, ...]

    with np.errstate(invalid="ignore", divide="ignore"):
        k_vec   = np.stack([KX, KY, KZ], axis=0)
        k_dot_u = np.sum(k_vec * U_hat, axis=0)
        proj    = k_dot_u / (K2 + 1e-30)
        U_hat   = U_hat - k_vec * proj[None, ...]

    u = np.fft.ifftn(U_hat[0]).real
    v = np.fft.ifftn(U_hat[1]).real
    w = np.fft.ifftn(U_hat[2]).real

    u -= u.mean()
    v -= v.mean()
    w -= w.mean()

    # Scale each component independently so that each one has the prescribed RMS.
    # This matches the intended interpretation of 0.1 * Delta_U per component.
    rms_u = math.sqrt((u**2).mean() + 1e-30)
    rms_v = math.sqrt((v**2).mean() + 1e-30)
    rms_w = math.sqrt((w**2).mean() + 1e-30)

    u *= amp_u / rms_u
    v *= amp_v / rms_v
    w *= amp_w / rms_w

    return u, v, w

#def params(Re, mu, delta, rho):
#
#    #Re = (U1 * rho * delta)/mu
#    u1 = (Re * mu)/(delta * rho)
#
#    #We = (rho * u1 * u1 * delta)/sigma
#    #sigma = (rho * u1 * u1 * delta)/We 
#
#    #print("--U1=",u1, "--Sigma=", sigma)
#    print("--U1=",u1)
#    return u1

def params(rho, delta_U, delta, Re):

    '''
    m      --> Ratio of mu2 / mu1
    rho    --> Density of fluid1
    delta  --> Initial momentum thickness
    Re     --> Reynolds number of fluid1
    We     --> Weber number as defined for fluid1
    '''

    #1.For Re
    #1.1 Compute mu_g
    mu = (rho * delta_U * delta) / Re
    
    print("--User entered delta_U: ", delta_U)
    print("--Computed mu1 & mu2: ", mu)
    mu1 = mu
    mu2 = mu

    return mu1, mu2

if __name__ == '__main__':
    from FPCSLpy.case import Case
    case = Case(path='./.')

    args = parse_args()
    nx_g, ny_g, nz_g = args.nx_g, args.ny_g, args.nz_g
    nxsd, nysd, nzsd = args.nxsd, args.nysd, args.nzsd
    amp              = args.amp

    dx = (2. * np.pi) / nx_g
    dy = (2. * np.pi) / ny_g
    dz = (2. * np.pi) / nz_g

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

    #Parameters
    rho1, rho2 = 1., 1.
    #mu changes!
    mu1 , mu2  = 1., 1.
    U1  , U2   = 1., 0.
    V1  , V2   = 0., 0.
    W1  , W2   = 0., 0.
    yloc = np.pi
    epsilon = 0.51 * case.grid['dx']
    delta   = 2. * np.pi/100.

    Re      = args.Re
    delta_U = args.delta_U
    sigma   = 0

    mu1, mu2 = params(rho1, delta_U, delta, Re)
    U1 =  0.5 * delta_U
    U2 = -0.5 * delta_U

    #Create custom initial conditions
    time_step = 0
    print("--", end="")
    case.create_default_fields(time_step)
    custom = case.data[f'{time_step}']

    phi1  = np.ones_like(case.grid['ycs'])
    phi2  = np.ones_like(case.grid['ycs'])
    print("--Shape of phi1 and phi2: ", phi1.shape, phi2.shape)
    rho   = phi1 * rho1 + (1. - phi1) * rho2
    mu    = phi1 * mu1  + (1. - phi1) * mu2
    idx1  = np.where(case.grid['ycs'] - yloc>=0.)
    idx2  = np.where(case.grid['ycs'] - yloc<0.)

    c = initialize_1D.get_optimal_c(case.grid['ycs'], (2 * np.pi/nx_g), yloc, U1, delta_U, delta)
    print("--Computed c val: ", c)

    ones  = np.ones_like(case.grid['ycs'])
    umean = np.zeros_like(case.grid['ycs'])
    u     = np.zeros_like(case.grid['ycs'])

    #umean = delta_U * 0.5 * np.tanh((case.grid['ycs'] - yloc) / (c * delta)) 
    umean = delta_U * 0.5 * np.tanh((case.grid['ycs'] - yloc) / (2 * delta)) 
    vmean = phi1 * V1 + (1. - phi1) * V2
    wmean = phi1 * W1 + (1. - phi1) * W2

    #Generating perturbations according to a specified spectrum
    unoise, vnoise, wnoise = generate_perturbations(nz_g, ny_g, nx_g, dz, dy, dx, amp, delta, delta_U)

    #Adding noise to mean
    u = umean + unoise
    v = vmean + vnoise
    w = wmean + wnoise

    u1_t = np.mean(umean, axis=(0,2))
    #alpha = np.mean(case.data[f'{time_step}']['phi_1'], axis=(0,2))
    alpha = np.mean(phi2, axis=(0,2))
    print("--Normalized momentum thickness: ", initialize_1D.momentum_thickness(U1, u1_t, ((2 * np.pi) / ny_g), delta_U) / delta)

    print("--U min & max val: ", u.min(), u.max())
    print("--V min & max val: ", v.min(), v.max())
    print("--W min & max val: ", w.min(), w.max())

    idx = int(case.grid['nx']/2.)
    fig, ax1 = plt.subplots(figsize=(3,3))
    ax1.plot(phi1[idx,:,2], case.grid['ycs'][idx,:,2], 'k')
    ax1.set(
        xlabel = r'$\phi_1$ [-]',
        ylabel = r'$y$ [m]',
    )
    ax2 = ax1.twiny()
    ax2.plot(umean[idx,:,2], case.grid['ycs'][idx,:,2], 'r--')
    ax2.plot(u[idx,:,2], case.grid['ycs'][idx,:,2], 'g--')
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
    ax2.plot(vmean[idx,:,2], case.grid['ycs'][idx,:,2], 'r--')
    ax2.plot(v[idx,:,2], case.grid['ycs'][idx,:,2], 'g--')
    ax2.set(
        xlabel = r'$v$ [m/s]',
    )
    fig.tight_layout()
    fig.savefig(f'./profiles_v.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4,3))
    cs = ax.pcolormesh(case.grid['xcs'][:,:,2], case.grid['ycs'][:,:,2], u[:,:,2], cmap='inferno')
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label(r'$u_1$ [m/s]')
    ax.set(
        xlabel = r'$x$ [m/s]',
        ylabel = r'$y$ [m/s]',
    )
    fig.tight_layout()
    fig.savefig(f'./noise.png', dpi=300)
    plt.close(fig)

    custom['u']     = u
    custom['v']     = v
    custom['w']     = w
    custom['phi_1'] = phi1
    custom['phi_2'] = phi2
    custom['rho']   = rho
    custom['p']     = ones
    custom['mu']    = mu
    print("--", end="")
    case.write_custom_fields(time_step, custom, to_interpolate=True)

