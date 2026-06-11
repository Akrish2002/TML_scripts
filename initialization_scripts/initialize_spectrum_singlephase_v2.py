import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    parser.add_argument('--m',       type=float, required=True)

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

def params(rho, delta_U, delta, Re, m):                                      
                                                                                
    '''                                                                         
    m      --> Ratio of mu2 / mu1                                               
    rho    --> Density of fluid1                                                
    delta  --> Initial momentum thickness                                       
    Re     --> Reynolds number of fluid1                                        
    '''                                                                         
                                                                                
    Re1     = Re                                                                
    rho1    = rho                                                               
                                                                                
    #1.For Re                                                                   
    #1.1 Compute mu_g                                                           
    mu1 = (rho1 * delta_U * delta) / Re1                                        
                                                                                
    #1.2 Compute mu_l                                                           
    mu2 = m * mu1                                                               
                                                                                
    print("--User entered delta_U: ", delta_U)                                            
    print("--Therefore U1 and U2: ", 0.5 * delta_U)                                            
    print("--Computed mu1, mu2: ", mu1, mu2)                                    
                                                                                
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
    m       = args.m
    sigma   = 0

    mu1, mu2 = params(rho1, delta_U, delta, Re, m)
    U1 =  0.5 * delta_U
    U2 = -0.5 * delta_U

    #Create custom initial conditions
    time_step = 0
    print("--", end="")
    case.create_default_fields(time_step)
    custom = case.data[f'{time_step}']

    profile = 0.5 * (1. + np.tanh((case.grid['ycs'] - yloc) / (2. * epsilon)))
    
    #Phi
    phi1  = np.ones_like(case.grid['ycs'])
    phi2  = np.zeros_like(case.grid['ycs'])
    print("--Shape of phi1 and phi2: ", phi1.shape, phi2.shape)

    #Physical params profile
    rho   = profile * rho1 + (1. - profile) * rho2
    mu    = profile * mu1  + (1. - profile) * mu2

    c = initialize_1D.get_optimal_c(case.grid['ycs'], (2 * np.pi/ny_g), yloc, U1, U2, delta_U, delta)
    print("--Computed c val: ", c)

    ones  = np.ones_like(case.grid['ycs'])
    umean = np.zeros_like(case.grid['ycs'])
    u     = np.zeros_like(case.grid['ycs'])

    umean = delta_U * 0.5 * np.tanh((case.grid['ycs'] - yloc) / (c * delta)) 
    vmean = phi1 * V1 + (1. - phi1) * V2
    wmean = phi1 * W1 + (1. - phi1) * W2

    #Generating perturbations according to a specified spectrum
    unoise, vnoise, wnoise = generate_perturbations(nz_g, ny_g, nx_g, dz, dy, dx, amp, delta, delta_U)

    #Adding noise to mean
    u = umean + unoise
    v = vmean + vnoise
    w = wmean + wnoise

    u1_t = np.mean(u, axis=(0,2))
    alpha = np.mean(phi2, axis=(0,2))
    print("--Normalized momentum thickness after adding pertubation: ", initialize_1D.momentum_thickness(U1, U2, u1_t, ((2 * np.pi) / ny_g), delta_U) / delta)

    print("--U min & max val: ", u.min(), u.max())
    print("--V min & max val: ", v.min(), v.max())
    print("--W min & max val: ", w.min(), w.max())

    iidx = int(case.grid['nx']/2.)
    kidx = int(case.grid['nz']/2.)
    umean_prof  = umean[iidx, :, kidx]
    u_prof      = u[iidx,   :, kidx]
    mu_prof     = mu[iidx,  :, kidx]
    rho_prof    = rho[iidx, :, kidx]

    fig, ax1 = plt.subplots(1, 2, figsize=(6,3))
    ax2 = [None, None]

    #1.
    ax1[0].plot(u_prof, case.grid['ycs'][iidx, :, kidx], 'g--')
    ax1[0].plot(umean_prof, case.grid['ycs'][iidx, :, kidx], 'b--')
    ax1[0].tick_params(axis='both', labelsize=6)
    ax1[0].set_xlabel(r'$u$ [m/s]', fontsize=6)
    ax1[0].set_ylabel(r'$y$ [m]', fontsize=6)

    #2.
    ax2[0] = ax1[0].twiny()
    ax2[0].plot(phi1[iidx,:,kidx], case.grid['ycs'][iidx,:,kidx], 'k')
    ax2[0].tick_params(axis='both', labelsize=6)
    ax2[0].set_xlabel(r'$\phi_1$ [-]', fontsize=6)
    
    #3.
    ax1[1].plot(mu_prof , case.grid['ycs'][iidx, :, kidx], 'r--')
    #ax1[1].ticklabel_format(axis='x', style='sci', scilimits=(-3, -3))
    ax1[1].axvline(mu2, color='k', linestyle='--', alpha=0.3)
    ax1[1].axvline(mu1, color='k', linestyle='--', alpha=0.3)
    ax1[1].tick_params(axis='both', labelsize=6)
    #ax1[1].xaxis.get_offset_text().set_fontsize(8)
    ax1[1].set_xlabel(r'$\mu$ [Pa-s]',fontsize=6)
    #ax1[1].set_ylabel(r'$y$ [m]', fontsize=6)

    #4.
    ax2[1] = ax1[1].twiny()
    ax2[1].plot(rho_prof, case.grid['ycs'][iidx, :, kidx], 'g--')
    ax2[1].tick_params(axis='both', labelsize=6)
    #formatter = mticker.ScalarFormatter(useMathText=True)
    #formatter.set_powerlimits((-3, -3))
    ax2[1].set_xlabel(r'$\rho$ [kg/m^3]',fontsize=6)

    fig.tight_layout()
    fig.savefig('./profiles_of_U_mu_rho.png', dpi=300)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(3,3))
    ax1.plot(phi1[iidx,:,kidx], case.grid['ycs'][iidx,:,kidx], 'k')
    ax1.set(
        xlabel = r'$\phi_1$ [-]',
        ylabel = r'$y$ [m]',
    )
    ax2 = ax1.twiny()
    ax2.plot(u[iidx,:,kidx], case.grid['ycs'][iidx,:,kidx], 'g--')
    ax2.plot(umean[iidx,:,kidx], case.grid['ycs'][iidx,:,kidx], 'r--')
    ax2.set(
        xlabel = r'$u$ [m/s]',
    )
    fig.tight_layout()
    fig.savefig(f'./profiles_u.png', dpi=300)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(3,3))
    ax1.plot(phi1[iidx,:,kidx], case.grid['ycs'][iidx,:,kidx], 'k')
    ax1.set(
        xlabel = r'$\phi_1$ [-]',
        ylabel = r'$y$ [m]',
    )
    ax2 = ax1.twiny()
    ax2.plot(vmean[iidx,:,kidx], case.grid['ycs'][iidx,:,kidx], 'r--')
    ax2.plot(v[iidx,:,kidx], case.grid['ycs'][iidx,:,kidx], 'g--')
    ax2.set(
        xlabel = r'$v$ [m/s]',
    )
    fig.tight_layout()
    fig.savefig(f'./profiles_v.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4,3))
    cs = ax.pcolormesh(case.grid['xcs'][:,:,kidx], case.grid['ycs'][:,:,kidx], u[:,:,kidx], cmap='inferno')
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

