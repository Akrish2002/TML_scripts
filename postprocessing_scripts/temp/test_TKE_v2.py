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


def read_field_large(case, field):

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
    block_org   = np.empty((a, b, c, nzsd, nxsd))
    block_avg   = np.empty((b))

    time_step = 2000
    for i in range(nxsd):
        for k in range(nzsd): 
            nxr, nyr, nzr = case.get_nxrnyrnzr_from_nr(filtered_block_list[i * nzsd + k])
            block         = case.read_block(time_step, nxr, nyr, nzr, to_read=[f'{field}'], to_interpolate=True)
            p     = block[f'{field}']
            #phi_1 = block['phi_1']
            
            block_org[..., k, i]      = p    
            #phi_1_block[..., k, i]  = phi_1
            
            del(block)

    #Averaging

    temp = block_org
    #Could you once again check why you are averaging out the alpha?
    #block_org = np.mean(temp, axis=(3, 4)) 
    block_avg = np.mean(temp, axis=(0, 2, 3, 4)) 
    #if(case.rank == 0): 
    #    print(temp.shape)
    #    print(block_org.shape)
    #    print(block_avg.shape)
    #    block_avg = block_avg[None, :, None]
    #    print(block_avg.shape)
    #    pprime = block_org - block_avg
    #    print(pprime.shape)
    #    pprime_ystack = np.mean(pprime, axis=(0,2))
    #    print(pprime_ystack.shape)
    #    print(pprime_ystack)

    #exit()    
 
    #Reshaping it into blocks of (512, 128, 512)
    #if(case.rank == 0): 
    #    print(block_org.shape)
    
    #How do I know the reshaping is done in coordinate(?) order
    block_org = block_org.reshape(a * nxsd, b, c * nzsd)
        
    temp_avg = block_avg[None, :, None]
    block_prime = block_org - temp_avg
    if(case.rank == 3):
        m = block_prime.mean(axis=(0, 2))
        rms = np.sqrt((block_prime**2).mean(axis=(0,2)))  # rms vs y
        rel = np.abs(m) / (rms + 1e-30)
        print(rel.max())
    exit()

    #Clumping for global u_profile
    parts     = case.comm.gather(block_org, root=0)
    parts_avg = case.comm.gather(block_avg, root=0)
    parts_prime = case.comm.gather(block_prime, root=0)
    
    if(case.rank == 0):
        global_field     = np.concatenate(parts, axis=1)
        global_field_avg = np.concatenate(parts_avg, axis=0)
        global_field_prime = np.concatenate(parts_prime, axis=1)
        
        print(global_field.shape)
        print(global_field_avg.shape)
        print(global_field_prime.shape)
        #print(np.mean(global_field_prime, axis=(0,2)))

        m = global_field_prime.mean(axis=(0, 2))
        rms = np.sqrt((global_field_prime**2).mean(axis=(0,2)))  # rms vs y
        rel = np.abs(m) / (rms + 1e-30)
        print(rel.max())

    
    else:
        global_field     = None
        global_field_avg = None
        global_field_prime = None


    exit()
    global_field     = case.comm.bcast(global_field, root=0)
    global_field_avg = case.comm.bcast(global_field_avg, root=0)
    global_field_prime = case.comm.bcast(global_field_prime, root=0)

    return global_field, global_field_avg, global_field_prime
    

def pressure_term(case):
    """
    Computing the pressure differential in the TKE budget

    """

    # 2. Pressure term
    # -\bar{(\frac{\partial p'}{\partial x_i}) * u''_i}
    
    # p' = p - \bar{p}
    p, p_bar, p_prime = read_field_large(case, 'p')
    #p_prime = p - p_bar                                                         #p' = p(x,y,z,t) - p(y,t)

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

