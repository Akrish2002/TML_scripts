import numpy as np
from mpi4py import MPI
from pathlib import Path
import re, pathlib
import matplotlib as mpl                                                        
import matplotlib.pyplot as plt
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

#Computing
def compute_reynolds_stresses(args):
    print("--Computing reynolds stresses!")

    T = TKE_Budget(args.case)
    T._time_step      = args.time_step
    T._stackdirection = args.stackdirection
    
    T.common_terms()
    T._option = 1
    T.miscellaneous()
    T.reynolds_stresses()

    if T._case.rank == 0:
        #Grepping the required data 
        U_l             = 0.
        U_g             = 3.1830988618379066
        print("-- Using hardcoded U_g and U_l values!")
        ny              = T._ny_g
        uprime_uprime   = T._uprime_uprime_global
        vprime_vprime   = T._vprime_vprime_global
        wprime_wprime   = T._wprime_wprime_global
        uprime_vprime   = T._uprime_vprime_global
        u_avg           = T._u_avg_global

        #Computing normalized time
        ctr_file        = os.path.join(args.case, "incompressible_tml.ctr")
        dt              = grep_ctr('dt', ctr_file)
        delta_ts        = (2. * np.pi) / 100.
        ts              = args.time_step
        t_normalized    = (ts * dt * U_g)/delta_ts

        #Generating y_grid
        dy = 2 * np.pi / ny
        print("--Using hardcoded domain size")
        #y_grid = np.arange(0, 2 * np.pi, step)
        y_grid = (np.arange(ny) + 0.5) * (dy)

        #Finding U(x, y_0.1 & 0.9, z) 
        U_01 = U_l + 0.1 * (U_g - U_l)
        U_09 = U_l + 0.9 * (U_g - U_l)

        idx = np.where((u_avg >= U_01) & (u_avg <= U_09))

        y_01 = y_grid[idx][0]
        y_09 = y_grid[idx][-1]

        delta = y_09 - y_01
        y_bar = 0.5 * (y_09 + y_01)
        #xi forms my new x
        xi = (y_grid - y_bar) / delta

        if(uprime_uprime.shape[0] != ny or 
           vprime_vprime.shape[0] != ny or 
           wprime_wprime.shape[0] != ny or 
           uprime_vprime.shape[0] != ny):
            #raise ValueError(f"uprime_uprime shape mismatch: got {uprime_uprime.shape}, expected ({ny},)")
            raise ValueError(f"Shape mismatch, please check generated dataset")
    
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"Reynolds_stresses_n{ny}_ts{int(args.time_step)}.npz"
    
        np.savez(
                    out_path,
                    case            =   str(Path(args.case).resolve()),

                    time_step       =   int(args.time_step),
                    t_normalized    =   np.float64(t_normalized),
                    ny              =   int(ny),

                    #Value stored, custom for each script
                    xi              =   xi.astype(np.float64),
                    uprime_uprime   =   uprime_uprime.astype(np.float64),
                    vprime_vprime   =   vprime_vprime.astype(np.float64),
                    wprime_wprime   =   wprime_wprime.astype(np.float64),
                    uprime_vprime   =   uprime_vprime.astype(np.float64)
                )
        print(f"[rank0] wrote {out_path} (ny={ny}, ts={args.time_step})")

#------------------------------------------------------------------------------

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

def load_npz_reynolds_stresses(path: str):
    d               = np.load(path, allow_pickle=True)
    case            = str(d["case"])

    time_step       = int(d["time_step"])
    t_normalized    = float(d["t_normalized"])
    ny              = int(d["ny"])

    xi              = d["xi"].astype(np.float64)
    uprime_uprime   = d["uprime_uprime"].astype(np.float64)
    vprime_vprime   = d["vprime_vprime"].astype(np.float64)
    wprime_wprime   = d["wprime_wprime"].astype(np.float64)
    uprime_vprime   = d["uprime_vprime"].astype(np.float64)

    print("uprime_uprime shape: ", uprime_uprime.shape)
    print("xi shape: ", xi.shape)

    if(uprime_uprime.ndim != 1 or uprime_uprime.shape != xi.shape or
       vprime_vprime.ndim != 1 or vprime_vprime.shape != xi.shape or
       wprime_wprime.ndim != 1 or wprime_wprime.shape != xi.shape or
       uprime_vprime.ndim != 1 or uprime_vprime.shape != xi.shape):
        raise ValueError(f"Size error while loading dataset, please check the generated dataset!")

    return case, t_normalized, ny, xi, \
           uprime_uprime,              \
           vprime_vprime,              \
           wprime_wprime,              \
           uprime_vprime

def plot_reynolds_stresses(args):
    entries = [load_npz_reynolds_stresses(f) for f in args.inputs]                                
    case    = [entry[0] for entry in entries]
                                                                                
    #Paper-style plot                                                           
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]), dpi=150)       
    ax = fig.add_subplot(111)                                                   
    dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]

    for idx, (case, t_normalized, ny, xi, \
              uprime_uprime,              \
              vprime_vprime,              \
              wprime_wprime,              \
              uprime_vprime) in enumerate(entries):

        lab = (
                args.labels[idx]
                if args.labels and len(args.labels) == len(entries)
                #else f"{Path(case).name} ({ny}$^3$)"
                else f"{ny}$^3$, t*={t_normalized:.2f}"
              )

        #Zoom mask in this
        x = xi
        y_list = [
            [uprime_uprime,
             vprime_vprime,
             wprime_wprime,
             uprime_vprime],
            [r"$\overline{u'u'}$",
             r"$\overline{v'v'}$",
             r"$\overline{w'w'}$",
             r"$\overline{u'v'}$"]
        ]
        y = y_list[0][args.component - 1]

        if args.zoom:
            x1 = 0 - args.zoom_window
            x2 = 0 + args.zoom_window
            m = (x >= x1) & (x <= x2)
            ax.plot(x[m], y[m], color="r", linestyle=dash_cycle[idx % len(dash_cycle)],
                    linewidth=1.2, label=lab)

        else:
            ax.plot(x, y, color="r", linestyle=dash_cycle[idx % len(dash_cycle)],
                    linewidth=1.2, label=lab)


    #Labels
    ax.set_ylabel(y_list[1][args.component - 1])
    ax.set_xlabel(r"$\xi$")
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
        ax.set_xlim(0 - args.zoom_window, 0 + args.zoom_window)

    apply_paper_style(ax)
    ax.legend(loc="best", frameon=False)
    ax.legend(fontsize=6)
    fig.tight_layout(pad=1.0)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
