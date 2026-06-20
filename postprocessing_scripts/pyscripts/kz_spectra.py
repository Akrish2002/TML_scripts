import numpy as np
from mpi4py import MPI
from pathlib import Path
import matplotlib as mpl                                                        
import matplotlib.pyplot as plt
import re, pathlib
import os

from pyscripts.test_TKE_vGPT_v3 import TKE_Budget

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

#Computing
def compute_spectra(args):
    T = TKE_Budget(args.case)
    T._time_step      = args.time_step
    T._stackdirection = args.stackdirection
    
    T.common_terms()
    T.compute_spectra_along_kx()

    if T._case.rank == 0:
        #Grepping the required data 
        E_kz = T._E_kz_global
        kz   = T._kz_positive_global

        ny   = T._ny_g
        y_max = T._ymax
        y = (np.arange(ny) + 0.5) * (y_max / ny)  

        #Computing normalized time
        U_l             = 0.
        U_g             = 3.1830988618379066
        ctr_file        = os.path.join(args.case, "incompressible_tml.ctr")
        dt              = grep_ctr('dt', ctr_file)
        delta_ts        = (2. * np.pi) / 100.
        ts              = args.time_step
        t_normalized    = (ts * dt * U_g)/delta_ts

        #if E_kx.ndim != 1 or E_kx.shape[0] != ny:
        if E_kz.shape[0] != ny:
            raise ValueError(f"E_kz shape mismatch: got {E_kz.shape}, ezpected ({ny},)")
    
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"Ez_spectra_n{ny}_ts{int(args.time_step)}.npz"

        np.savez(
                    out_path,
                    case            =   str(Path(args.case).resolve()),

                    time_step       =   int(args.time_step),
                    t_normalized    =   np.float64(t_normalized),
                    ny              =   int(ny),

                    y               =   y.astype(np.float64),
                    kx              =   kx.astype(np.float64),
                    E_kx            =   E_kx.astype(np.complex128)
                    
                )
        print(f"[rank0] wrote {out_path} (ny={ny}, ts={args.time_step})")

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
    d    = np.load(path, allow_pickle=True)
    case = str(d["case"])

    t_normalized    = float(d["t_normalized"])
    y               = d["y"].astype(np.float64)
    ny              = int(d["ny"])

    kx   = d["kx"].astype(np.float64)
    E_kx = d["E_kx"].astype(np.complex128)

    print("E_kx shape: ", E_kx.shape)
    print("y shape: ", y.shape)
    print("kx shape: ", kx.shape)

    print("E_kx[1] shape: ", E_kx.shape[1])
    print("kx[1] shape: ", kx.shape[0])
    if E_kx.ndim == 1 or E_kx.shape[1] != kx.shape[0]:
        raise ValueError(f"Size error, please check the generated dataset!")

    print("-- Returning spectra only at the centerplane!")
    return case, t_normalized, ny, kx, E_kx[len(E_kx) // 2]

def plot_spectra(args):
    #Loads only the E_kx along the centerline
    entries = [load_npz_spectra(f) for f in args.inputs]                                
    case    = [entry[0] for entry in entries]
                                                                                
    #Paper-style plot                                                           
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]), dpi=150)       
    ax = fig.add_subplot(111)                                                   
    dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]

    for idx, (case, t_normalized, ny, kx, E_kx) in enumerate(entries):
        lab = (
                args.labels[idx]
                if args.labels and len(args.labels) == len(entries)
                else f"{ny}$^3$, t*={t_normalized:.2f}"
              )

        #Zoom mask in this
        ax.legend(fontsize=8)
        if args.zoom:
            print("--Zoom has not been implemented for spectra plotting")
            ax.loglog(kx[1:], E_kx[1:], color="r", 
                    linestyle=dash_cycle[idx % len(dash_cycle)], linewidth=1.2, label=lab)

        else:
            ax.loglog(kx[1:], E_kx[1:], color="r", 
                    linestyle=dash_cycle[idx % len(dash_cycle)], linewidth=1.2, label=lab)

    # --- Two reference slope ---
    k_ref = kx[1:]                 # avoid k=0
    k0 = k_ref[5]                  # pick a reference point (adjust index if needed)
    E0 = E_kx[5].real             # corresponding energy
    c = 0.75
    
    E_ref_1 = E0 * (k_ref / k0)**(-5.0/3.0)
    E_ref_2 = E0 * (k_ref / k0)**(-10.0/3.0) * np.exp(c)
    
    ax.loglog(k_ref, E_ref_1, 'k--', linewidth=1.2, label=r"$k^{-5/3}$")
    ax.loglog(k_ref, E_ref_2, 'k-.', linewidth=1.2, label=r"$k^{-10/3}$")


    #Labels
    ax.set_ylabel("$E(k)_{x}$")
    ax.set_xlabel("$k_{x}$")
    #To have path of run being used
    p = Path(case)
    short = Path(*p.parts[-2:])
    fig.text(
        0.98, 0.01, short,
        ha="right",
        va="bottom",
        fontsize=5
    )

    if args.zoom:
        print("--Zoom has not been implemented for spectra plotting!")

    apply_paper_style(ax)
    ax.legend(loc="best", frameon=False)
    ax.legend(fontsize=6)
    fig.tight_layout(pad=1.0)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    

