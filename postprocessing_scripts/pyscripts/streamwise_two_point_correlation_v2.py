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
def compute_streamwise_two_point_correlation(args):
    T = TKE_Budget(args.case)
    T._time_step      = args.time_step
    T._stackdirection = args.stackdirection
    
    T.common_terms()
    T.compute_streamwise_two_point_correlation()

    if T._case.rank == 0:
        #Grepping the required data 
        Ruu_x       = T._Ruu_x_global
        Rvv_x       = T._Rvv_x_global
        Rww_x       = T._Rww_x_global
        Rphi2phi2_x = T._Rphi2phi2_x_global

        ny    = T._ny_g
        nx    = T._nx_g
        y_max = T._ymax
        y     = (np.arange(ny) + 0.5) * (y_max / ny)  
        rx    = np.arange(nx) * T._dx

        #Computing normalized time
        U_l             = 0.
        U_g             = 3.1830988618379066
        ctr_file        = os.path.join(args.case, "incompressible_tml.ctr")
        dt              = grep_ctr('dt', ctr_file)
        delta_ts        = (2. * np.pi) / 100.
        ts              = args.time_step
        t_normalized    = (ts * dt * U_g)/delta_ts

        if Ruu_x.shape[0] != ny:
            raise ValueError(f"Ruu_x shape mismatch: got {Ruu_x.shape}, expected ({ny},)")
        if Rvv_x.shape[0] != ny:
            raise ValueError(f"Rvv_x shape mismatch: got {Rvv_x.shape}, expected ({ny},)")
        if Rww_x.shape[0] != ny:
            raise ValueError(f"Rww_x shape mismatch: got {Rww_x.shape}, expected ({ny},)")
        if Rphi2phi2_x.shape[0] != ny:
            raise ValueError(f"Rphi2phi2_x shape mismatch: got {Rphi2phi2_x.shape}, expected ({ny},)")
    
        out_path = Path(args.output_path)
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"streamwise_two_point_correlation_n{ny}_ts{int(args.time_step)}.npz"

        np.savez(
                    out_path,
                    case                =   str(Path(args.case).resolve()),

                    time_step           =   int(args.time_step),
                    t_normalized        =   np.float64(t_normalized),
                    ny                  =   int(ny),
                    nx                  =   int(nx),
                    y_max               =   np.float64(y_max),

                    y                   =   y.astype(np.float64),
                    rx                  =   rx.astype(np.float64),

                    Ruu_x          =   Ruu_x.astype(np.float64),
                    Rvv_x          =   Rvv_x.astype(np.float64),
                    Rww_x          =   Rww_x.astype(np.float64),
                    Rphi2phi2_x    =   Rphi2phi2_x.astype(np.float64)
                    
                )
        print(f"[rank0] wrote {out_path} (ny={ny}, ts={args.time_step})\n")


#Plotting
def ensemble_average_correlation_entries(entries):
    (
        case0,
        t0,
        ny0,
        y0,
        y_max0,
        rz0,
        Ruu0,
        Rvv0,
        Rww0,
        Rphi2phi20,
    ) = entries[0]

    t_values = []

    Ruu_list = []
    Rvv_list = []
    Rww_list = []
    Rphi2phi2_list = []

    for entry in entries:
        (
            case,
            t_normalized,
            ny,
            y,
            y_max,
            rz,
            Ruu_z,
            Rvv_z,
            Rww_z,
            Rphi2phi2_z,
        ) = entry

        if ny != ny0:
            raise ValueError(f"ny mismatch: got {ny}, expected {ny0}")

        if not np.isclose(y_max, y_max0):
            raise ValueError(f"y_max mismatch: got {y_max}, expected {y_max0}")

        if not np.allclose(y, y0):
            raise ValueError("y grid mismatch between correlation files")

        if not np.allclose(rz, rz0):
            raise ValueError("rz grid mismatch between correlation files")

        Ruu_list.append(Ruu_z)
        Rvv_list.append(Rvv_z)
        Rww_list.append(Rww_z)
        Rphi2phi2_list.append(Rphi2phi2_z)

        t_values.append(t_normalized)

    Ruu_avg = np.mean(np.stack(Ruu_list, axis=0), axis=0)
    Rvv_avg = np.mean(np.stack(Rvv_list, axis=0), axis=0)
    Rww_avg = np.mean(np.stack(Rww_list, axis=0), axis=0)
    Rphi2phi2_avg = np.mean(np.stack(Rphi2phi2_list, axis=0), axis=0)

    return (
        case0 + " | ensemble",
        float(np.mean(t_values)),
        ny0,
        y0,
        y_max0,
        rz0,
        Ruu_avg,
        Rvv_avg,
        Rww_avg,
        Rphi2phi2_avg,
    )

def load_npz_streamwise_two_point_correlation(path: str):
    #For normalization
    delta_ts = (2 * np.pi) / 100

    d    = np.load(path, allow_pickle=True)
    case = str(d["case"])

    t_normalized      = float(d["t_normalized"])
    y                 = d["y"].astype(np.float64)
    ny                = int(d["ny"])
    y_max             = float(d["y_max"])

    rx              = d["rx"].astype(np.float64)
    rx_normalized   = rx / delta_ts
    Ruu_x           = d["Ruu_x"].astype(np.float64)
    Rvv_x           = d["Rvv_x"].astype(np.float64)
    Rww_x           = d["Rww_x"].astype(np.float64)
    Rphi2phi2_x     = d["Rphi2phi2_x"].astype(np.float64)

    print("Ruu_x shape: ", Ruu_x.shape)
    print("Rvv_x shape: ", Rvv_x.shape)
    print("Rww_x shape: ", Rww_x.shape)
    print("Rphi2phi2_x shape: ", Rphi2phi2_x.shape)
    print("y shape: ", y.shape)
    print("rx shape: ", rx.shape)

    if Ruu_x.ndim == 1 or Ruu_x.shape[1] != rx.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")
    if Rvv_x.ndim == 1 or Rvv_x.shape[1] != rx.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")
    if Rww_x.ndim == 1 or Rww_x.shape[1] != rx.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")
    if Rphi2phi2_x.ndim == 1 or Rphi2phi2_x.shape[1] != rx.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")

    return case, t_normalized, ny, y, y_max, rx_normalized, Ruu_x, Rvv_x, Rww_x, Rphi2phi2_x


def plot_streamwise_two_point_correlation(args):
    #Loads only the correlation along the requested y-locations
    entries = [load_npz_streamwise_two_point_correlation(f) for f in args.inputs]
    
    if getattr(args, "ensemble_average", False):
        entries = [ensemble_average_correlation_entries(entries)]
    cases   = [entry[0] for entry in entries]
                                                                                
    #Paper-style plot                                                           
    paper_style() 
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]), dpi=150)       
    ax = fig.add_subplot(111)                                                   
    dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]

    components = get_components(args)
    component_map = {
        1: ("R_{uu}", "Ruu_x"),
        2: ("R_{vv}", "Rvv_x"),
        3: ("R_{ww}", "Rww_x"),
        4: ("R_{\phi_2\phi_2}", "Rphi2phi2_x"),
    }

    curve_count = 0

    for idx, (case, t_normalized, ny, y, y_max, rx_normalized, Ruu_x, Rvv_x, Rww_x, Rphi2phi2_x) in enumerate(entries):
        case_lab = (
                args.labels[idx]
                if args.labels and len(args.labels) == len(entries)
                #else f"{ny}$^3$, t*={t_normalized:.2f}"
                else f"{t_normalized:.2f}"
              )

        y_delta_multiples = get_y_delta_multiples(args)
        y_indices, y_selected = get_y_indices(y, y_max, y_delta_multiples)

        correlation_map = {
            "Ruu_x"       : Ruu_x,
            "Rvv_x"       : Rvv_x,
            "Rww_x"       : Rww_x,
            "Rphi2phi2_x" : Rphi2phi2_x,
        }

        for m_idx, m in zip(y_indices, y_delta_multiples):
            for component in components:
                if component not in component_map:
                    raise ValueError("Component must be one of 1, 2, 3, 4")

                component_label, component_key = component_map[component]
                #To avoid accidently plotting phi2 correlation during single phase
                if(component_key == "Rphi2phi2_x" and args.singlephase):
                    continue

                R_x = correlation_map[component_key][m_idx, :]

                #lab = f"{case_lab}, {component_label}, y-yc={m:g}$\\delta_{{\\theta,0}}$"
                if(args.ensemble_average):
                    lab = f"${component_label}$"
                else:
                    lab = f"$t^* = {case_lab}$, ${component_label}$"

                half_idx = len(rx_normalized) // 2 + 1
                ax.plot(rx_normalized[:half_idx], R_x[:half_idx], color="r", linestyle=dash_cycle[curve_count % len(dash_cycle)], label=lab)
                curve_count = curve_count + 1


    #Horizontal line at y = 0
    plt.axhline(y=0, linestyle="--", color="k", linewidth=0.5, alpha=0.5)
    #Labels
    ax.set_ylabel("$R(r_x, z) / R(0, x)$")
    ax.set_xlabel("$r_x / \delta_{0}$")
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
