import numpy as np
import h5py
from mpi4py import MPI
from pathlib import Path
import re, pathlib

from pyscripts.test_TKE_vGPT_v1 import TKE_Budget

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

    return fs, step, ls


def total_dissipation_of_TKE(args):

    # Grep the start, step interval, and end timestep from the case directory
    (start_ts, step_ts, end_ts) = grep_timestep(args.case)

    # Build the full list of timesteps to iterate over
    time_steps = []
    for i in range(start_ts, end_ts + step_ts, step_ts):
        time_steps.append(i)

    # Initialise a temporary TKE_Budget object to read grid metadata
    # before entering the main loop
    T = TKE_Budget(args.case)

    # Debug: print timestep information on rank 0 only
    if T._case.rank == 0:
        print(start_ts)
        print(step_ts)
        print(end_ts)

    # Read the number of grid points and compute the grid spacing in y
    ny = T._ny_g
    dy = 2 * np.pi / ny

    # Determine output path and create parent directories if they do not exist
    out_path = Path(args.out).with_suffix(".h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # On rank 0, initialise the HDF5 file and create the resizable datasets
    # up front so that every iteration can simply append to them.
    # The datasets are created with maxshape=(None,) which allows them
    # to be extended along axis 0 one entry at a time.
    if T._case.rank == 0:
        hdf5_file = h5py.File(out_path, "w")

        # Store the case path as a file-level attribute for provenance
        hdf5_file.attrs["case"] = str(Path(args.case).resolve())

        # Create a group to hold all output datasets
        grp = hdf5_file.create_group("data")

        # Create the time_steps dataset: starts empty, grows by one each iteration
        ts_dataset = grp.create_dataset(
            "time_steps",
            shape   = (0,),
            maxshape= (None,),
            dtype   = np.float64
        )

        # Create the total dissipation dataset: same shape policy as time_steps
        diss_dataset = grp.create_dataset(
            "total_dissipation_of_TKE",
            shape   = (0,),
            maxshape= (None,),
            dtype   = np.float64
        )

    # Clean up the temporary TKE_Budget object used for metadata
    del T

    # Main loop: process one timestep at a time so that results are written
    # to disk immediately and nothing is held in memory across iterations
    for time_step in time_steps:

        # Instantiate TKE_Budget for the current timestep
        T                 = TKE_Budget(args.case)
        T._time_step      = time_step
        T._stackdirection = args.stackdirection

        # Compute the common intermediate terms required for all budget terms
        T.common_terms()

        # Compute the dissipation term of the TKE budget
        T.dissipation()

        # Only rank 0 holds the global assembled field and performs I/O
        if T._case.rank == 0:

            # Integrate the dissipation profile over y using the trapezoidal rule
            # to obtain a single scalar value for this timestep
            scalar_dissipation = np.trapezoid(T._dissipation_global, dx=dy)

            # Determine the new length after appending one element
            current_length = ts_dataset.shape[0]
            new_length     = current_length + 1

            # Extend both datasets by one slot along axis 0
            ts_dataset.resize(new_length, axis=0)
            diss_dataset.resize(new_length, axis=0)

            # Write the current timestep and its dissipation value into the new slot
            ts_dataset[current_length]   = np.float64(time_step)
            diss_dataset[current_length] = np.float64(scalar_dissipation)

            # Flush the HDF5 file to disk so the data is safe even if the job dies
            hdf5_file.flush()

            print(f"[rank0] wrote ts={time_step}, dissipation={scalar_dissipation:.6e} -> {out_path}")

        # Free the TKE_Budget object before starting the next iteration
        del T

    # Close the HDF5 file cleanly after all timesteps have been processed.
    # This is only done on rank 0 because only rank 0 opened the file.
    if T._case.rank == 0:
        hdf5_file.close()
        print(f"[rank0] finished. Wrote {len(time_steps)} timesteps to {out_path}")
