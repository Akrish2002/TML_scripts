import numpy as np
from mpi4py import MPI
from pathlib import Path
import re, pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from .postprocess_parallelcore_v1 import grep_timestep
from .postprocess_parallelcore_v1 import case_update
from .postprocess_parallelcore_v1 import split_timestep_over_cores


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
    n = float(m.group(1)) if m else None

    return n

#Computing
def compute_reynolds_number(args):
    """ Compute Reynolds number based on momentum thickness.
    
    Args:
        args : Parsed command line arguments
    
    Return:
        Saves .npz file containing normalized time and Re_theta(t)
    """

    # Grepping the required data
    U_l      = 0.
    U_g      = 3.1830988618379066
    delta_ts = (2. * np.pi) / 100.
    delta_U  = U_g - U_l

    ctr_file = os.path.join(args.case, "incompressible_tml.ctr")
    dt       = grep_ctr("dt", ctr_file)
    #Grepping physical params of gas to compute \nu
    rho1     = grep_ctr("rho1o", ctr_file)
    mu1      = grep_ctr("mu1", ctr_file)
    nu_g     = mu1 / rho1

    #The postprocessing utilities use the case directory as their working path.
    current_working_directory = os.getcwd()
    os.chdir(args.case)

    try:
        #The postprocessing utilities use the case directory as their working path.
        (start_ts, step_ts, end_ts) = grep_timestep()

        (nx_g, ny_g, nz_g,
         dx, dy, dz,
         case) = case_update("incompressible_tml.ctr")
        ny_g_half = int(ny_g / 2)

        time_steps           = []
        t_normalized         = []
        reynolds_number      = []
        momentum_thickness   = []

        for time_step in range(start_ts, end_ts, step_ts):
            (mt, pt, mixt) = split_timestep_over_cores(delta_ts, U_g, dt, delta_U,

                                                       nx_g, ny_g, nz_g,
                                                       ny_g_half,
                                                       dx, dy, dz,
                                                       case,

                                                       time_step)

            # The imported momentum thickness is normalized by delta_ts.
            delta_theta = mt * delta_ts
            #Computing the reynolds number
            Re_theta    = (delta_theta * delta_U) / nu_g 
            t_star      = (time_step * dt * U_g) / delta_ts

            time_steps.append(time_step)
            reynolds_number.append(Re_theta)
            t_normalized.append(t_star)

        if case.rank == 0:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez(
                        out_path,
                        case                = str(Path(args.case).resolve()),
                        nx                  = int(nx_g),

                        #Value stored, custom for each script
                        time_step           = np.asarray(time_steps, dtype=np.int64),
                        t_normalized        = np.asarray(t_normalized, dtype=np.float64),
                        reynolds_number     = np.asarray(reynolds_number, dtype=np.float64)
                    )
            print(f"[rank0] wrote {out_path} (nx={nx_g}, ny={ny_g}, nz={nz_g})")
    
    finally:
        os.chdir(current_working_directory)

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


def load_npz_reynolds_number(path: str):
    """ Load the Reynolds number dataset.
    
    Args:
        path (string) : Path to the .npz dataset
    
    Return:
        case            (string)      : Case path
        nx              (int)         : Grid size in x
        t_normalized    (ndarray)     : Normalized time array
        reynolds_number (ndarray)     : Reynolds number array
    """

    d = np.load(path, allow_pickle=True)

    case            = str(d["case"])
    nx              = int(d["nx"])
    t_normalized    = d["t_normalized"].astype(np.float64)
    reynolds_number = d["reynolds_number"].astype(np.float64)

    print("t_normalized shape: ", t_normalized.shape)
    print("reynolds_number shape: ", reynolds_number.shape)

    if(t_normalized.ndim != 1 or
       reynolds_number.ndim != 1 or
       t_normalized.shape != reynolds_number.shape):
        raise ValueError(f"Size error while loading dataset, please check the generated dataset!")

    return case, nx, t_normalized, reynolds_number


def plot_reynolds_number(args):
    """ Plot Reynolds number against normalized time.
    
    Args:
        args : Parsed command line arguments
    
    Return:
        Saves .png plot
    """

    entries = [load_npz_reynolds_number(f) for f in args.inputs]
    case    = entries[0][0]

    #Paper-style plot
    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]), dpi=150)
    ax = fig.add_subplot(111)
    dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]

    for idx, (case_i, nx, t_normalized, reynolds_number) in enumerate(entries):
        lab = (
                args.labels[idx]
                if args.labels and len(args.labels) == len(entries)
                else f"{nx}$^3$"
              )

        x = t_normalized
        y = reynolds_number

        ax.plot(x, y, linestyle=dash_cycle[idx % len(dash_cycle)],
                linewidth=1.2, label=lab)

    #Labels
    ax.set_xlabel(r"$t^*$")
    ax.set_ylabel(r"$Re_{\theta}$")

    #To have path of run being used
    p = Path(case)
    short = Path(*p.parts[-2:])
    fig.text(
        0.98, 0.01, short,
        ha="right",
        va="bottom",
        fontsize=5
    )

    apply_paper_style(ax)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout(pad=1.0)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
