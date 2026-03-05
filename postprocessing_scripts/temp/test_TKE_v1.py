import numpy as np
from mpi4py import MPI
import numpy as np
import os
import csv
import argparse
import re, pathlib
from pathlib import Path
import sys


def grep_ctr(st, ctr_file="../incompressible_tml.ctr"):
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


def case_update():
    #Hardcoded for 512
    from FPCSLpy.case import Case
    case = Case(path='/storage/home/hcoda1/5/agopalak33/abhijeet/13_TML/04_FrontierData/n512')

    #args = parse_args()
    #nx_g, ny_g = args.nx_g, args.ny_g
    nx_g = 512
    ny_g = 512
    nz_g = 512

    # Update default parameters
    to_update_parameters = case.parameters
    to_update_parameters['grid']['x_max'] = 2.*np.pi
    to_update_parameters['grid']['y_max'] = 2.*np.pi
    to_update_parameters['grid']['z_max'] = 2.*np.pi
    to_update_parameters['grid']['nx'] = nx_g
    to_update_parameters['grid']['ny'] = ny_g
    to_update_parameters['grid']['nz'] = nz_g
    to_update_parameters['simulation_parameters']['parallel']['nxsd'] = 8
    to_update_parameters['simulation_parameters']['parallel']['nysd'] = 4
    to_update_parameters['simulation_parameters']['parallel']['nzsd'] = 4
    to_update_parameters['simulation_parameters']['solvers']['incompressible'] = True

    # Update and check parameters   
    case.update_parameters(to_update_parameters)

    return case


def large_case_update():
    from FPCSLpy.case import LargeCase
    #case = LargeCase(path='/ccs/home/abhi/member-work/incompressible-tml/BaseCase/n512')
    case = LargeCase(path='../')

    nx_g = 512
    ny_g = 512
    nz_g = 512

    #Update default parameters
    to_update_parameters = case.parameters
    to_update_parameters['grid']['x_max']                                       = 2 * np.pi
    to_update_parameters['grid']['y_max']                                       = 2 * np.pi
    to_update_parameters['grid']['z_max']                                       = 2 * np.pi
    to_update_parameters['grid']['nx']                                          = nx_g
    to_update_parameters['grid']['ny']                                          = ny_g
    to_update_parameters['grid']['nz']                                          = nz_g
    to_update_parameters['simulation_parameters']['parallel']['nxsd']           = 8
    to_update_parameters['simulation_parameters']['parallel']['nysd']           = 4
    to_update_parameters['simulation_parameters']['parallel']['nzsd']           = 4
    to_update_parameters['simulation_parameters']['solvers']['incompressible']  = True

    #Update and check parameters
    case.update_parameters(to_update_parameters)

    return case


def read_field(case):

    i = 1000
    time_step = i
    time_steps = [time_step]
    case.read_time_steps(time_steps, to_interpolate=True)

    #u , v , w  = case.data[f'{time_step}']['uc'], \
    #             case.data[f'{time_step}']['vc'], \
    #             case.data[f'{time_step}']['wc']

    p = case.data[f'{time_step}']['p']
    print(p.shape())
    exit()
    
    return p


def read_field_large(case):

    case.distribute_block_list_axis_stack(1)
    filtered_block_list = case.filtered_rank_block_list
    #print(filtered_block_list)
    #exit()

    nxsd = int(grep_ctr("nxsd"))
    nysd = int(grep_ctr("nysd"))
    nzsd = int(grep_ctr("nzsd"))

    #nx_g = int(grep_ctr("nx_g"))
    #ny_g = int(grep_ctr("ny_g"))
    #nz_g = int(grep_ctr("nz_g"))

    nx_g = 512
    ny_g = 512
    nz_g = 512

    #Maybe change this from being a hardcoded size?
    a = int(nx_g / nxsd)
    b = int(ny_g / nysd)
    c = int(nz_g / nzsd)
    if(case.rank == 0):
        print(a, b, c)
    p_block     = np.empty((a, b, c, nzsd, nxsd))
    phi_1_block = np.empty((a, b, c, nzsd, nxsd))

    time_step = 2000
    for i in range(nxsd):
        for k in range(nzsd): 
            nxr, nyr, nzr = case.get_nxrnyrnzr_from_nr(filtered_block_list[i * nzsd + k])
            block         = case.read_block(time_step, nxr, nyr, nzr, to_read=['p', 'phi_1'], to_interpolate=True)
            p     = block['p']
            phi_1 = block['phi_1']
            
            p_block[..., k, i]      = p    
            phi_1_block[..., k, i]  = phi_1
            
            del(block)

    #Averaging
    #Could you once again check why you are averaging out the alpha?
    p_bar = np.mean(p_block, axis=(0, 2, 3, 4)) 
    phi_1 = np.mean(phi_1_block, axis=(0, 2, 3, 4)) 
    
    #Clumping for global u_profile
    parts_p     = case.comm.gather(p_bar, root=0)
    parts_phi_1 = case.comm.gather(phi_1, root=0)
    
    if(case.rank == 0):
        global_p_bar = np.concatenate(parts_p, axis=0)
        global_phi_1 = np.concatenate(parts_phi_1, axis=0)
        
        print(global_p_bar.shape)
    
    else:
        global_p_bar = None
        global_phi_1 = None


    exit()

    global_p_bar = case.comm.bcast(global_p_bar, root=0)
    global_phi_1 = case.comm.bcast(global_phi_bar, root=0)

    return global_p_bar
    

def pressure_term(case):
    """
    Computing the pressure differential in the TKE budget

    """

    # 2. Pressure term
    # -\bar{(\frac{\partial p'}{\partial x_i}) * u''_i}
    
    # p' = p - \bar{p}
    p = read_field_large(case)
    p_bar = np.mean(read_field_large(case), axis=(0, 2))
    p_prime = p - p_bar                                                         #p' = p(x,y,z,t) - p(y,t)

    # u''_i = u_i - \tilde{u_i}
    # \tilde{u_i} = \bar{\rho * u} / \rho
    phi_1 = read_data(phi_1)                                                    #This gives me phi_1 as a matrix(?), I can then construct rho from this
    rho = phi_1 * rho_1 + (1 - phi_1) * rho_2

    u = read_data(u)
    u_tilde = np.mean(rho * u) / np.mean(rho)
    u_double_prime = u - u_tilde
    

def main():
    #print("asdf")
    #case = case_update()
    case = large_case_update()
    pressure_term(case)


if __name__ == "__main__":
    main()

