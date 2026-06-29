import numpy as np
from mpi4py import MPI
from pathlib import Path
import matplotlib as mpl                                                        
import matplotlib.pyplot as plt
import re, pathlib
import os

from pyscripts.test_TKE_vGPT_v4 import TKE_Budget
from pyscripts.plot_style import paper_style 


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


def grep_timestep(path = "."):                                                  
    """ Grepping time steps to calculate first step, step and last step         
                                                                                
    Args:                                                                       
                                                                                
    Return:                                                                     
                                                                                
    """                                                                         
                                                                                
    root = Path(path)                                                           
    nums = []                                                                   
                                                                                
    for p in root.iterdir():                                                    
        m = re.fullmatch(r"time_step-(\d+)", p.name)                            
        if m:                                                                   
            nums.append(int(m.group(1)))                                        
                                                                                
    nums.sort()                                                                 
    if nums:                                                                    
        fs, step, ls = min(nums), nums[1] - nums[0], max(nums)                  


def get_y_delta_multiples(args, fallback=None):
    if hasattr(args, "y_delta_multiples") and args.y_delta_multiples is not None:
        y_delta_multiples = np.asarray(args.y_delta_multiples, dtype=np.float64)
    elif fallback is not None:
        y_delta_multiples = np.asarray(fallback, dtype=np.float64)
    else:
        y_delta_multiples = np.asarray([0.0], dtype=np.float64)

    return y_delta_multiples


def get_y_indices(y, y_max, y_delta_multiples):
    delta_theta0 = (2. * np.pi) / 100.
    yc = 0.5 * y_max

    y_indices = []
    y_selected = []

    for m in y_delta_multiples:
        y_target = yc + m * delta_theta0
        idx = int(np.argmin(np.abs(y - y_target)))
        y_indices.append(idx)
        y_selected.append(y[idx])

    return np.asarray(y_indices, dtype=np.int64), np.asarray(y_selected, dtype=np.float64)


def get_components(args):
    if hasattr(args, "spectra_components") and args.spectra_components is not None:
        components = list(args.spectra_components)
    else:
        components = [1, 2, 3, 4]

    return components


#Computing
def compute_spectra(args):
    T = TKE_Budget(args.case)
    T._time_step      = args.time_step
    T._stackdirection = args.stackdirection
    
    T.common_terms()
    T.compute_spectra_along_kz()

    if T._case.rank == 0:
        #Grepping the required data 
        E_uu_kz  = T._E_uu_kz_global
        E_vv_kz  = T._E_vv_kz_global
        E_ww_kz  = T._E_ww_kz_global
        E_TKE_kz = T._E_TKE_kz_global
        kz       = T._kz_positive_global

        ny    = T._ny_g
        y_max = T._ymax
        y     = (np.arange(ny) + 0.5) * (y_max / ny)  

        #Computing normalized time
        U_l             = 0.
        U_g             = 3.1830988618379066
        ctr_file        = os.path.join(args.case, "incompressible_tml.ctr")
        dt              = grep_ctr('dt', ctr_file)
        delta_ts        = (2. * np.pi) / 100.
        ts              = args.time_step
        t_normalized    = (ts * dt * U_g)/delta_ts

        if E_uu_kz.shape[0] != ny:
            raise ValueError(f"E_uu_kz shape mismatch: got {E_uu_kz.shape}, expected ({ny},)")
        if E_vv_kz.shape[0] != ny:
            raise ValueError(f"E_vv_kz shape mismatch: got {E_vv_kz.shape}, expected ({ny},)")
        if E_ww_kz.shape[0] != ny:
            raise ValueError(f"E_ww_kz shape mismatch: got {E_ww_kz.shape}, expected ({ny},)")
        if E_TKE_kz.shape[0] != ny:
            raise ValueError(f"E_TKE_kz shape mismatch: got {E_TKE_kz.shape}, expected ({ny},)")
    
        out_path = Path(args.output_path)
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"kz_spectra_n{ny}_ts{int(args.time_step)}.npz"

        np.savez(
                    out_path,
                    case                =   str(Path(args.case).resolve()),

                    time_step           =   int(args.time_step),
                    t_normalized        =   np.float64(t_normalized),
                    ny                  =   int(ny),
                    y_max               =   np.float64(y_max),

                    y                   =   y.astype(np.float64),

                    kz                  =   kz.astype(np.float64),
                    E_uu_kz             =   E_uu_kz.astype(np.float64),
                    E_vv_kz             =   E_vv_kz.astype(np.float64),
                    E_ww_kz             =   E_ww_kz.astype(np.float64),
                    E_TKE_kz            =   E_TKE_kz.astype(np.float64)
                    
                )
        print(f"[rank0] wrote {out_path} (ny={ny}, ts={args.time_step})\n")


#Plotting
def apply_paper_style(ax):
    # light dotted grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, color="0.55")

    # black frame
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("k")

    # tick style
    ax.tick_params(direction="out", length=4, width=1.0, colors="k")


def load_npz_spectra(path: str):
    rho_g   = 1.0
    U_l     = 0.0
    U_g     = 3.1830988618379066
    delta_U = U_g - U_l
    delta0 = (2.0 * np.pi) / 100.0

    d    = np.load(path, allow_pickle=True)
    case = str(d["case"])

    t_normalized      = float(d["t_normalized"])
    y                 = d["y"].astype(np.float64)
    ny                = int(d["ny"])
    y_max             = float(d["y_max"])

    kz       = d["kz"].astype(np.float64)
    kz_normalized = kz * delta0
    E_uu_kz  = d["E_uu_kz"].astype(np.float64) / (delta_U**2 * delta0)
    E_vv_kz  = d["E_vv_kz"].astype(np.float64) / (delta_U**2 * delta0)
    E_ww_kz  = d["E_ww_kz"].astype(np.float64) / (delta_U**2 * delta0)
    E_TKE_kz = d["E_TKE_kz"].astype(np.float64) / (delta_U**2 * delta0)

    print("E_uu_kz shape: ", E_uu_kz.shape)
    print("E_vv_kz shape: ", E_vv_kz.shape)
    print("E_ww_kz shape: ", E_ww_kz.shape)
    print("E_TKE_kz shape: ", E_TKE_kz.shape)
    print("y shape: ", y.shape)
    print("kz shape: ", kz.shape)

    if E_uu_kz.ndim == 1 or E_uu_kz.shape[1] != kz.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")
    if E_vv_kz.ndim == 1 or E_vv_kz.shape[1] != kz.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")
    if E_ww_kz.ndim == 1 or E_ww_kz.shape[1] != kz.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")
    if E_TKE_kz.ndim == 1 or E_TKE_kz.shape[1] != kz.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")

    return case, t_normalized, ny, y, y_max, kz_normalized, E_uu_kz, E_vv_kz, E_ww_kz, E_TKE_kz


def plot_spectra(args):
    #Loads only the E_kz along the requested y-locations
    entries = [load_npz_spectra(f) for f in args.inputs]                                
    cases   = [entry[0] for entry in entries]
                                                                                
    #Paper-style plot                                                           
    paper_style()
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]), dpi=150)       
    ax = fig.add_subplot(111)                                                   
    dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]

    components = get_components(args)
    component_map = {
        1: ("E_{uu}", "E_uu_kz"),
        2: ("E_{vv}", "E_vv_kz"),
        3: ("E_{ww}", "E_ww_kz"),
        4: ("E_{u_iu_i}", "E_TKE_kz"),
    }

    curve_count = 0
    k_ref_for_slope = None
    E_ref_for_slope = None

    for idx, (case, t_normalized, ny, y, y_max, kz, E_uu_kz, E_vv_kz, E_ww_kz, E_TKE_kz) in enumerate(entries):
        case_lab = (
                args.labels[idx]
                if args.labels and len(args.labels) == len(entries)
                else f"$t^* = {t_normalized:.2f}$"
              )

        y_delta_multiples = get_y_delta_multiples(args)
        y_indices, y_selected = get_y_indices(y, y_max, y_delta_multiples)

        spectra_map = {
            "E_uu_kz"  : E_uu_kz,
            "E_vv_kz"  : E_vv_kz,
            "E_ww_kz"  : E_ww_kz,
            "E_TKE_kz" : E_TKE_kz,
        }

        for m_idx, m in zip(y_indices, y_delta_multiples):
            for component in components:
                if component not in component_map:
                    raise ValueError("Component must be one of 1, 2, 3, 4")

                component_label, component_key = component_map[component]
                E_kz = spectra_map[component_key][m_idx, :]

                #lab = f"{case_lab}, {component_label}, y-yc={m:g}$\\delta_{{\\theta,0}}$"
                lab = f"{case_lab}, ${component_label}$"

                k_plot = kz[1:]
                E_plot = E_kz[1:]
                
                E_floor = 1e-6
                valid = E_plot >= E_floor
                
                ax.loglog(k_plot[valid], E_plot[valid], color="r", 
                        linestyle=dash_cycle[curve_count % len(dash_cycle)], label=lab)

                if k_ref_for_slope is None:
                    k_ref_for_slope = kz[1:]
                    E_ref_for_slope = E_kz[1:]

                curve_count = curve_count + 1

    # --- Two reference slope ---
    if k_ref_for_slope is not None and len(k_ref_for_slope) > 6:
        k_ref = k_ref_for_slope
        k0 = k_ref[5]
        E0 = E_ref_for_slope[5]
        c = 0.75
        
        E_ref_1 = E0 * (k_ref / k0)**(-5.0/3.0)
        #E_ref_2 = E0 * (k_ref / k0)**(-10.0/3.0) * np.exp(c)
        
        ax.loglog(k_ref, E_ref_1, 'k--', linewidth=1.2, label=r"$k^{-5/3}$")
        #ax.loglog(k_ref, E_ref_2, 'k-.', linewidth=1.2, label=r"$k^{-10/3}$")

    #Labels
    ax.set_ylabel("$E(k)_{z} / (\Delta U^2 \delta_0)$")
    ax.set_xlabel("$k_{z}\delta_0$")
    #To have path of run being used
    p = Path(cases[-1])
    short = Path(*p.parts[-2:])
    fig.text(
        0.98, 0.01, short,
        ha="right",
        va="bottom",
        fontsize=5
    )

    ax.legend()
    fig.tight_layout(pad=1.0)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
