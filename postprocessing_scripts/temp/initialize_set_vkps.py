import os
import cfg
import math
import numpy as np
import cantera as ct

# =========================================================
# Derived dimensional parameters
# =========================================================

gamma_ref = cfg.gamma
Rg_ref = cfg.Rgas
p0 = cfg.p_ref

T_top = cfg.Theta_T_top * cfg.T_ref
T_bot = cfg.Theta_T_bot * cfg.T_ref

a_top = math.sqrt(gamma_ref * Rg_ref * T_top)
a_bot = math.sqrt(gamma_ref * Rg_ref * T_bot)

# Magnitudes from Mach numbers
U_top_mag = cfg.M_top * a_top
U_bot_mag = cfg.M_bot * a_bot

# Temporal mixing layer: opposite-signed streams
U_top = +U_top_mag          # hot air, +x
U_bot = -U_bot_mag          # cold H2/air, -x

# Domain dimensions
Lx = cfg.Lx_over_delta0 * cfg.delta0
Ly = cfg.Ly_over_delta0 * cfg.delta0
Lz = cfg.Lz_over_delta0 * cfg.delta0

Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

x = (np.arange(Nx) + 0.5) * dx
y = (np.arange(Ny) + 0.5) * dy
z = (np.arange(Nz) + 0.5) * dz

Z, Yc, Xc = np.meshgrid(z, y, x, indexing="ij")

# =========================================================
# Cantera setup
# =========================================================

gas = ct.Solution(cfg.chem_mech)
Ns = gas.n_species

# Species indices for diagnostics
H2_IND = gas.species_index("H2") if "H2" in gas.species_names else None
H2O_IND = gas.species_index("H2O") if "H2O" in gas.species_names else None

def idx(species_name: str) -> int:
    return gas.species_index(species_name)

# =========================================================
# State indices
# =========================================================

RHO, RHOU, RHOV, RHOW, RHOE = 0, 1, 2, 3, 4
RHOY0 = 5
nvar = RHOY0 + Ns

# =========================================================
# Initialization: Supersonic H2–Air temporal mixing layer
# =========================================================

# Shear-layer center
y0 = 0.5 * Ly

# Tanh blend across y
s = 2.0 * (Yc - y0) / max(1e-12, cfg.delta0)
blend = 0.5 * (1.0 + np.tanh(s))  # 0 bottom -> 1 top

# Base stream velocities and temperatures
u_base = (1.0 - blend) * U_bot + blend * U_top
v_base = np.zeros_like(u_base)
w_base = np.zeros_like(u_base)

T_field = (1.0 - blend) * T_bot + blend * T_top

# Density from reference ideal gas (only for init)
rho_field = p0 / (Rg_ref * T_field)

# --- Species composition ---

# Top stream: hot air
gas.TPX = T_top, p0, cfg.top_comp_air
Y_top = gas.Y.copy()  # (Ns,)

# Bottom stream: H2 + air, with 6% H2 by volume
X_H2 = cfg.X_H2_vol
X_air = 1.0 - X_H2

gas.TPX = T_bot, p0, cfg.bottom_comp_air
X_air_vec = gas.X.copy()

X_bottom = X_air_vec * X_air
X_bottom[idx("H2")] += X_H2
X_bottom /= X_bottom.sum()

gas.TPX = T_bot, p0, X_bottom
Y_bot = gas.Y.copy()

# Blend species across the shear layer
Y_field = (1.0 - blend)[None, ...] * Y_bot[:, None, None, None] + \
          blend[None, ...] * Y_top[:, None, None, None]

# =========================================================
# 3D von Kármán–Pao-like perturbations
# =========================================================

def generate_vk_perturbations(Nz, Ny, Nx, dz, dy, dx, amp):
    """
    Generate a 3D divergence-free velocity perturbation field (u', v', w')
    with a von Kármán–Pao-like energy spectrum.
    """
    kx = 2.0 * math.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * math.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2.0 * math.pi * np.fft.fftfreq(Nz, d=dz)

    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    k = np.sqrt(K2)

    k0 = 2.0 * math.pi / max(Lx, Ly, Lz)
    k_eta = 0.5 * (Nx + Ny + Nz) * 2.0 * math.pi / max(Lx, Ly, Lz)

    E_k = np.zeros_like(k)
    mask = k > 0
    kk = k[mask]
    E_k[mask] = (kk**4) / (k0**2 + kk**2)**(17.0/6.0) * np.exp(-2.0 * (kk/k_eta)**2)

    random_phase = (np.random.normal(size=(3, Nz, Ny, Nx)) +
                    1j * np.random.normal(size=(3, Nz, Ny, Nx)))

    A = np.sqrt(E_k + 1e-30)
    U_hat = random_phase * A[None, ...]

    with np.errstate(invalid="ignore", divide="ignore"):
        k_vec = np.stack([KX, KY, KZ], axis=0)
        k_dot_u = np.sum(k_vec * U_hat, axis=0)
        proj = k_dot_u / (K2 + 1e-30)
        U_hat = U_hat - k_vec * proj[None, ...]

    u = np.fft.ifftn(U_hat[0]).real
    v = np.fft.ifftn(U_hat[1]).real
    w = np.fft.ifftn(U_hat[2]).real

    u -= u.mean()
    v -= v.mean()
    w -= w.mean()

    rms = math.sqrt((u**2 + v**2 + w**2).mean() + 1e-30)
    u *= amp / rms
    v *= amp / rms
    w *= amp / rms

    return u, v, w

pert_amp = cfg.pert_amp * abs(U_top - U_bot)
du, dv, dw = generate_vk_perturbations(Nz, Ny, Nx, dz, dy, dx, pert_amp)

u = u_base + du
v = v_base + dv
w = w_base + dw
