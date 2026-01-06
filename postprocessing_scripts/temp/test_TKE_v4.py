from mpi4py import MPI
import numpy as np
import os
import csv
import argparse
import re, pathlib
import matplotlib.pyplot as plt
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

    dx = (case.parameters['grid']['x_max'] - case.parameters['grid']['x_min']) / case.parameters['grid']['nx']
    dy = (case.parameters['grid']['y_max'] - case.parameters['grid']['y_min']) / case.parameters['grid']['ny']
    dz = (case.parameters['grid']['z_max'] - case.parameters['grid']['z_min']) / case.parameters['grid']['nz']

    return case, dx, dy, dz


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

    nxsd = int(grep_ctr("nxsd"))
    nysd = int(grep_ctr("nysd"))
    nzsd = int(grep_ctr("nzsd"))

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
            block_org[..., k, i]     = block[f'{field}']
            
            #p     = block[f'{field}']
            #block_org[..., k, i]      = p    
            
            del(block)

    #Averaging
    temp = block_org
    block_avg = np.mean(temp, axis=(0, 2, 3, 4)) 

    #How do I know the reshaping is done in coordinate(?) order
    #I think it isnt, so I have to transpose
    #GPT helped me with the transpose idea, but there is a stack answer disucssing this
    #, which I must read
    #GPT_Idea
    slab = block_org.transpose(4, 0, 1, 3, 2)
    block_org = slab.reshape(a * nxsd, b, c * nzsd)
    temp_avg = block_avg[None, :, None]
    block_prime = block_org - temp_avg

    #if(case.rank == 3):
    #    m = block_prime.mean(axis=(0, 2))
    #    rms = np.sqrt((block_prime**2).mean(axis=(0,2)))  # rms vs y
    #    rel = np.abs(m) / (rms + 1e-30)
    #    print(rel.max())
    #exit()

    #Clumping for global u_profile
    parts     = case.comm.gather(block_org, root=0)
    parts_avg = case.comm.gather(block_avg, root=0)
    parts_prime = case.comm.gather(block_prime, root=0)
    
    if(case.rank == 0):
        global_field     = np.concatenate(parts, axis=1)
        global_field_avg = np.concatenate(parts_avg, axis=0)
        global_field_prime = np.concatenate(parts_prime, axis=1)
        
        #print(global_field.shape)
        #print(global_field_avg.shape)
        #print(global_field_prime.shape)

        #m = global_field_prime.mean(axis=(0, 2))
        #rms = np.sqrt((global_field_prime**2).mean(axis=(0,2)))  # rms vs y
        #rel = np.abs(m) / (rms + 1e-30)
        #print(rel.max())

    
    else:
        global_field     = None
        global_field_avg = None
        global_field_prime = None


    #exit()
    global_field     = case.comm.bcast(global_field, root=0)
    global_field_avg = case.comm.bcast(global_field_avg, root=0)
    global_field_prime = case.comm.bcast(global_field_prime, root=0)

    return global_field, global_field_avg, global_field_prime
    

def first_order_derivative(field, direction, dx, periodic=False):
    #1. Construct matrix to populate with derivatives
    #2. Construct code to compute derivatives
    #3. Keep check for edges values since they cannot be computed using central schemes
    
    #1.
    Derivative = np.empty_like(field, dtype=float)
    #periodic = True
    
    #2.
    length = field.shape[direction - 1]
    
    axis = direction - 1
    
    if(periodic):
      Derivative = (np.roll(field, -1, axis) - np.roll(field, 1, axis)) / (2*dx)
    
    else:
      D_center_1 = [slice(None)] * field.ndim
      D_center_1[axis] = slice(2, None)
    
      D_center_2 = [slice(None)] * field.ndim
      D_center_2[axis] = slice(None, -2)
    
      D_top_edge_1 = [slice(None)] * field.ndim
      D_top_edge_1[axis] = 0
    
      D_top_edge_2 = [slice(None)] * field.ndim
      D_top_edge_2[axis] = 1
    
      D_top_edge_3 = [slice(None)] * field.ndim
      D_top_edge_3[axis] = 2
    
      D_bottom_edge_1 = [slice(None)] * field.ndim
      D_bottom_edge_1[axis] = -1
    
      D_bottom_edge_2 = [slice(None)] * field.ndim
      D_bottom_edge_2[axis] = -2
    
      D_bottom_edge_3 = [slice(None)] * field.ndim
      D_bottom_edge_3[axis] = -3
    
      idx_center      = [slice(None)] * field.ndim
      idx_center[axis] = slice(1, -1)
      idx_top_edge    = [slice(None)] * field.ndim
      idx_top_edge[axis] = 0
      idx_bottom_edge = [slice(None)] * field.ndim
      idx_bottom_edge[axis] = -1
    
      Derivative[tuple(idx_center)] = (field[tuple(D_center_1)] - field[tuple(D_center_2)]) / (2*dx)
      Derivative[tuple(idx_top_edge)] = (-3*field[tuple(D_top_edge_1)] + 4*field[tuple(D_top_edge_2)] - field[tuple(D_top_edge_3)]) / (2*dx)
      Derivative[tuple(idx_bottom_edge)] = (3*field[tuple(D_bottom_edge_1)] - 4*field[tuple(D_bottom_edge_2)] + field[tuple(D_bottom_edge_3)]) / (2*dx)
    

    return Derivative
            

def pressure_term(case, dx, dy, dz):
    """
    Computing the pressure differential in the TKE budget

    """

    rho1 = grep_ctr("rho1o")
    rho2 = grep_ctr("rho2o")
    delta = [dx, dy, dz]

    # 2. Pressure term
    # -\bar{(\frac{\partial p'}{\partial x_i}) * u''_i}
    
    # p' = p - \bar{p}
    _, _, p_prime = read_field_large(case, 'p')                                 #p' = p(x,y,z,t) - p(y,t)

    #Since in this particular case I have equal density and incompressible
    _, _, u_double_prime = read_field_large(case, 'u')
    _, _, v_double_prime = read_field_large(case, 'v')
    _, _, w_double_prime = read_field_large(case, 'w')

    #Compute instantaneous differential term
    dpprime_dx1 = first_order_derivative(p_prime, 1, delta[0], True)
    dpprime_dx2 = first_order_derivative(p_prime, 2, delta[1], False)
    dpprime_dx3 = first_order_derivative(p_prime, 3, delta[2], True)
        
    #Form the summed term
    q = -(u_double_prime * dpprime_dx1 + v_double_prime * dpprime_dx2 + w_double_prime * dpprime_dx3)

    #Average
    q = np.mean(q, axis=(0, 2))
    
    if(case.rank == 0):
        plot(dy, q)

    return q
     

def plot(dy, field):
    #y = np.linspace(0, 2*np.pi, len(field), endpoint=False, dtype=np.float64)

    y_max = 2*np.pi
    y_min = 0

    ny = len(field)
    dy = (y_max - y_min) / ny
    y = y_min + (np.arange(ny) + 0.5)*dy
    
    plt.figure()
    plt.plot(y, field)
    plt.xlabel("y")
    plt.ylabel(r"$\overline{-u_i''\,\partial_i p'}_{xz}$")
    plt.title("Pressure term profile (plane-averaged)")
    plt.grid(True)
    plt.tight_layout()
   
    plt.savefig("pressure_term_profile.png", dpi=300)
    plt.close() 
    

def main():
    #print("asdf")
    #case = case_update()
    case, dx, dy, dz = large_case_update()
    Pressure_Term = pressure_term(case, dx, dy, dz)
    print(Pressure_Term.shape)
    


if __name__ == "__main__":
    main()

