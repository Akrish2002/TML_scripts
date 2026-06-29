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
        #idx = int(np.argmin(np.abs(y - y_target)))
        idx = 512
        y_indices.append(idx)
        y_selected.append(y[idx])

    return np.asarray(y_indices, dtype=np.int64), np.asarray(y_selected, dtype=np.float64)


def get_components(args):
    if hasattr(args, "twopoint_correlation_components") and args.twopoint_correlation_components is not None:
        components = list(args.twopoint_correlation_components)
    else:
        components = [1, 2, 3, 4]

    return components


#Computing
def compute_spanwise_two_point_correlation(args):
    T = TKE_Budget(args.case)
    T._time_step      = args.time_step
    T._stackdirection = args.stackdirection
    
    T.common_terms()
    T.compute_spanwise_two_point_correlation()

    if T._case.rank == 0:
        #Grepping the required data 
        Ruu_z       = T._Ruu_z_global
        Rvv_z       = T._Rvv_z_global
        Rww_z       = T._Rww_z_global
        Rphi2phi2_z = T._Rphi2phi2_z_global

        ny    = T._ny_g
        nz    = T._nz_g
        y_max = T._ymax
        y     = (np.arange(ny) + 0.5) * (y_max / ny)  
        rz    = np.arange(nz) * T._dz

        #Computing normalized time
        U_l             = 0.
        U_g             = 3.1830988618379066
        ctr_file        = os.path.join(args.case, "incompressible_tml.ctr")
        dt              = grep_ctr('dt', ctr_file)
        delta_ts        = (2. * np.pi) / 100.
        ts              = args.time_step
        t_normalized    = (ts * dt * U_g)/delta_ts

        if Ruu_z.shape[0] != ny:
            raise ValueError(f"Ruu_z shape mismatch: got {Ruu_z.shape}, expected ({ny},)")
        if Rvv_z.shape[0] != ny:
            raise ValueError(f"Rvv_z shape mismatch: got {Rvv_z.shape}, expected ({ny},)")
        if Rww_z.shape[0] != ny:
            raise ValueError(f"Rww_z shape mismatch: got {Rww_z.shape}, expected ({ny},)")
        if Rphi2phi2_z.shape[0] != ny:
            raise ValueError(f"Rphi2phi2_z shape mismatch: got {Rphi2phi2_z.shape}, expected ({ny},)")
    
        out_path = Path(args.output_path)
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"spanwise_two_point_correlation_n{ny}_ts{int(args.time_step)}.npz"

        np.savez(
                    out_path,
                    case                =   str(Path(args.case).resolve()),

                    time_step           =   int(args.time_step),
                    t_normalized        =   np.float64(t_normalized),
                    ny                  =   int(ny),
                    nz                  =   int(nz),
                    y_max               =   np.float64(y_max),

                    y                   =   y.astype(np.float64),
                    rz                  =   rz.astype(np.float64),

                    Ruu_z               =   Ruu_z.astype(np.float64),
                    Rvv_z               =   Rvv_z.astype(np.float64),
                    Rww_z               =   Rww_z.astype(np.float64),
                    Rphi2phi2_z         =   Rphi2phi2_z.astype(np.float64)
                    
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


def load_npz_spanwise_two_point_correlation(path: str):
    #For normalization
    delta_ts = (2 * np.pi) / 100

    d    = np.load(path, allow_pickle=True)
    case = str(d["case"])

    t_normalized      = float(d["t_normalized"])
    y                 = d["y"].astype(np.float64)
    ny                = int(d["ny"])
    y_max             = float(d["y_max"])

    rz                =     d["rz"].astype(np.float64)
    rz_normalized     =     rz / delta_ts
    Ruu_z             =     d["Ruu_z"].astype(np.float64)
    Rvv_z             =     d["Rvv_z"].astype(np.float64)
    Rww_z             =     d["Rww_z"].astype(np.float64)
    Rphi2phi2_z       =     d["Rphi2phi2_z"].astype(np.float64)

    print("Ruu_z shape: ", Ruu_z.shape)
    print("Rvv_z shape: ", Rvv_z.shape)
    print("Rww_z shape: ", Rww_z.shape)
    print("Rphi2phi2_z shape: ", Rphi2phi2_z.shape)
    print("y shape: ", y.shape)
    print("rz shape: ", rz.shape)

    if Ruu_z.ndim == 1 or Ruu_z.shape[1] != rz.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")
    if Rvv_z.ndim == 1 or Rvv_z.shape[1] != rz.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")
    if Rww_z.ndim == 1 or Rww_z.shape[1] != rz.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")
    if Rphi2phi2_z.ndim == 1 or Rphi2phi2_z.shape[1] != rz.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")

    return case, t_normalized, ny, y, y_max, rz_normalized, Ruu_z, Rvv_z, Rww_z, Rphi2phi2_z


def plot_spanwise_two_point_correlation(args):
    #Loads only the correlation along the requested y-locations
    entries = [load_npz_spanwise_two_point_correlation(f) for f in args.inputs]                                
    cases   = [entry[0] for entry in entries]
                                                                                
    #Paper-style plot                                                           
    paper_style()
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]), dpi=150)       
    ax = fig.add_subplot(111)                                                   
    dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]

    components = get_components(args)
    component_map = {
        1: ("R_{uu}", "Ruu_z"),
        2: ("R_{vv}", "Rvv_z"),
        3: ("R_{ww}", "Rww_z"),
        4: ("R_{\phi_2\phi_2}", "Rphi2phi2_z"),
    }

    curve_count = 0

    for idx, (case, t_normalized, ny, y, y_max, rz_normalized, Ruu_z, Rvv_z, Rww_z, Rphi2phi2_z) in enumerate(entries):
        case_lab = (
                args.labels[idx]
                if args.labels and len(args.labels) == len(entries)
                #else f"{ny}$^3$, t*={t_normalized:.2f}"
                #else f"t*={t_normalized:.2f}"
                else f"{t_normalized:.2f}"
              )

        y_delta_multiples = get_y_delta_multiples(args)
        y_indices, y_selected = get_y_indices(y, y_max, y_delta_multiples)

        correlation_map = {
            "Ruu_z"       : Ruu_z,
            "Rvv_z"       : Rvv_z,
            "Rww_z"       : Rww_z,
            "Rphi2phi2_z" : Rphi2phi2_z,
        }

        for m_idx, m in zip(y_indices, y_delta_multiples):
            for component in components:
                if component not in component_map:
                    raise ValueError("Component must be one of 1, 2, 3, 4")

                component_label, component_key = component_map[component]
                R_z = correlation_map[component_key][m_idx, :]

                #lab = f"{case_lab}, {component_label}, y-yc={m:g}$\\delta_{{\\theta,0}}$"
                lab = f"$t^* = {case_lab}$, ${component_label}$"

                half_idx = len(rz_normalized) // 2 + 1
                ax.plot(rz_normalized[:half_idx], R_z[:half_idx], color="r", 
                        linestyle=dash_cycle[curve_count % len(dash_cycle)], label=lab)

                curve_count = curve_count + 1

    #Horizontal line at y = 0
    plt.axhline(y=0, linestyle="--", color="k", linewidth=0.5, alpha=0.5)
    #Labels
    ax.set_ylabel("$R(r_z, z) / R(0, z)$")
    ax.set_xlabel("$r_z / \delta_{0}$")
    #To have path of run being used
    p = Path(cases[-1])
    short = Path(*p.parts[-2:])
    fig.text(
        0.98, 0.01, short,
        ha="right",
        va="bottom",
        fontsize=5
    )

    #apply_paper_style(ax)
    ax.legend()
    ax.legend()
    fig.tight_layout(pad=1.0)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
