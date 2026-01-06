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


def read_field_large(case, field, phi=False):

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
    block_org       = np.empty((a, b, c, nzsd, nxsd))
    block_org_phi   = np.empty((a, b, c, nzsd, nxsd))
    block_avg   = np.empty((b))
    block_avg_phi   = np.empty((b))

    time_step = 7000
    for i in range(nxsd):
        for k in range(nzsd): 
            nxr, nyr, nzr = case.get_nxrnyrnzr_from_nr(filtered_block_list[i * nzsd + k])

            block         = case.read_block(time_step, nxr, nyr, nzr, to_read=[f'{field}', 'phi_1'], to_interpolate=True)
            block_org[..., k, i]     = block[f'{field}']
            block_org_phi[..., k, i] = block['phi_1']
            
            #p     = block[f'{field}']
            #block_org[..., k, i]      = p    
            
            del(block)

    #Averaging
    block_avg = np.mean(block_org, axis=(0, 2, 3, 4)) 
    block_avg_phi = np.mean(block_org_phi, axis=(0, 2, 3, 4)) 

    #How do I know the reshaping is done in coordinate(?) order
    #I think it isnt, so I have to transpose
    #GPT helped me with the transpose idea, but there is a stack answer disucssing this
    #, which I must read
    #GPT_Idea
    slab = block_org.transpose(4, 0, 1, 3, 2)
    slab_phi = block_org_phi.transpose(4, 0, 1, 3, 2)

    block_org = slab.reshape(a * nxsd, b, c * nzsd)
    block_org_phi = slab_phi.reshape(a * nxsd, b, c * nzsd)

    #temp_avg = block_avg[None, :, None]
    block_avg = block_avg[None, :, None]
    block_prime = block_org - block_avg

    #if(case.rank == 3):
    #    m = block_prime.mean(axis=(0, 2))
    #    rms = np.sqrt((block_prime**2).mean(axis=(0,2)))  # rms vs y
    #    rel = np.abs(m) / (rms + 1e-30)
    #    print(rel.max())
    #exit()

    #Returning global_field, global_field_avg, global_field_prime
    if(phi == True):
        return block_org, block_avg, block_prime, block_org_phi, block_avg_phi
    else:
        return block_org, block_avg, block_prime
    

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

            
#2nd Term
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

    '''
    Since in this particular case I have equal density and incompressible
    '''
    _, _, u_double_prime = read_field_large(case, 'u')
    _, _, v_double_prime = read_field_large(case, 'v')
    _, _, w_double_prime = read_field_large(case, 'w')

    '''
    Compute instantaneous differential term
    '''
    dpprime_dx1 = first_order_derivative(p_prime, 1, delta[0], True)
    dpprime_dx2 = first_order_derivative(p_prime, 2, delta[1], False)
    dpprime_dx3 = first_order_derivative(p_prime, 3, delta[2], True)
        
    '''
    Form the summed term
    '''
    q = -(u_double_prime * dpprime_dx1 + v_double_prime * dpprime_dx2 + w_double_prime * dpprime_dx3)

    #Average
    #Clump all then stack here instead?
    #if(case.rank == 0):
    #    print(q.shape)
    #exit()

    q_parts     = case.comm.gather(q, root=0)
    
    if(case.rank == 0):
        q_global     = np.concatenate(q_parts, axis=1)
        '''
        This peforms the overbar from the eqn
        '''
        q_global = np.mean(q_global, axis=(0, 2))
        plot(dy, q_global)

        #m = global_field_prime.mean(axis=(0, 2))
        #rms = np.sqrt((global_field_prime**2).mean(axis=(0,2)))  # rms vs y
        #rel = np.abs(m) / (rms + 1e-30)
        #print(rel.max())
    
    else:
        q_global     = None


#3rd Term
def turbulent_diffusion(case, dx, dy, dz):
    #u', v', w'
    _, _, u_double_prime = read_field_large(case, 'u')
    _, _, v_double_prime = read_field_large(case, 'v')
    _, _, w_double_prime = read_field_large(case, 'w')

    vel_double_prime_square_sum = u_double_prime * u_double_prime  + \
                                  v_double_prime * v_double_prime  + \
                                  w_double_prime * w_double_prime 
 
    
    '''
    Constructing the terms inside the differential
    '''
    #t1 = np.mean(u_double_prime * vel_double_prime_square_sum, axis=(0,2))
    t2 = 0.5 * (np.mean(v_double_prime * vel_double_prime_square_sum, axis=(0,2)))
    #t3 = np.mean(w_double_prime * vel_double_prime_square_sum, axis=(0,2))

    #If I do not do this I get an error in the derivative?
    #t1 = t1[None, :, None]
    t2 = t2[None, :, None]
    #t3 = t3[None, :, None]

    '''
    Finding the differential
    '''
    #dt1prime_dx1 = first_order_derivative(t1, 1, dx, True) 
    dt2prime_dx2 = first_order_derivative(t2, 2, dy, False) 
    #dt3prime_dx3 = first_order_derivative(t3, 3, dz, True) 

    '''
    Forming the summed term
    '''
    #q = dt1prime_dx1 - dt2prime_dx2 - dt3prime_dx3
    #q = -dt2prime_dx2
    q = dt2prime_dx2
    q_parts     = case.comm.gather(q, root=0)
    
    if(case.rank == 0):
        q_global     = np.concatenate(q_parts, axis=1)
        
        #print(q_global.shape)

        '''
        This is just to convert it to (ny,) there is no change in value
        '''
        q_global = np.mean(q_global, axis=(0, 2))
        plot(dy, q_global)

        #m = global_field_prime.mean(axis=(0, 2))
        #rms = np.sqrt((global_field_prime**2).mean(axis=(0,2)))  # rms vs y
        #rel = np.abs(m) / (rms + 1e-30)
        #print(rel.max())
    
    else:
        q_global     = None


#1st Term
def advection(case, dx, dy, dz):
    '''
    Getting u', v', w' and Avg terms. I should be getting Favre avg, but in this particular
    case I am getting reynolds avg, since it is equal density and incompressible?
    '''
    _, u_avg, u_double_prime = read_field_large(case, 'u')
    _, v_avg, v_double_prime = read_field_large(case, 'v')
    _, w_avg, w_double_prime = read_field_large(case, 'w')
    

    '''
    Finding k
    I have to make it as [None, :, None], so that I can multiply? Yea I need to
    do that to broadcast them
    '''
    k = np.mean(0.5 * (u_double_prime * u_double_prime  + \
                       v_double_prime * v_double_prime  + \
                       w_double_prime * w_double_prime), axis=(0,2))
    k = k[None, :, None]

    '''
    Constructing the terms inside each differential
    '''
    #t1 = u_avg * k
    t2 = v_avg * k
    #t3 = w_avg * k

    
    '''
    Finding the differential
    '''
    #dt1_dx1 = first_order_derivative(t1, 1, dx, True) 
    dt2_dx2 = first_order_derivative(t2, 2, dy, False) 
    #dt3_dx3 = first_order_derivative(t3, 3, dz, True) 

    '''
    Forming the summed term
    '''
    #q = d_dx1 + d_dx2 + d_dx3
    q = -dt2_dx2
    q_parts     = case.comm.gather(q, root=0)
    
    if(case.rank == 0):
        q_global     = np.concatenate(q_parts, axis=1)
        
        #print(q_global.shape)
        #q = np.mean(q_global, axis=(0, 2))

        '''
        This is just to convert it to (ny,) there is no change in value
        '''
        q_global = np.mean(q_global, axis=(0, 2))
        plot(dy, q_global)

        #m = global_field_prime.mean(axis=(0, 2))
        #rms = np.sqrt((global_field_prime**2).mean(axis=(0,2)))  # rms vs y
        #rel = np.abs(m) / (rms + 1e-30)
        #print(rel.max())
    
    else:
        q_global     = None


#5th Term
def dissipiation(case, dx, dy, dz):
    '''
    Have to compute the viscosity term    
    '''
    mu_1 = 1e-3
    mu_2 = 5e-2

    _, u_avg, u_double_prime, phi_1, _    = read_field_large(case, 'u', True)
    _, v_avg, v_double_prime            = read_field_large(case, 'v', False)
    _, w_avg, w_double_prime            = read_field_large(case, 'w', False)

    mu = phi_1 * mu_1 + (1 - phi_1) * mu_2
    #mu_avg = np.mean((phi_1 * mu_1 + (1 - phi_1) * mu_2), axis=(0, 2))
    #mu_avg = mu_avg[None, :, None]


    '''
    All terms
    '''
    dudoubleprime_dx = first_order_derivative(u_double_prime, 1, dx, True)
    dudoubleprime_dy = first_order_derivative(u_double_prime, 2, dy, False)
    dudoubleprime_dz = first_order_derivative(u_double_prime, 3, dz, True)

    dvdoubleprime_dx = first_order_derivative(v_double_prime, 1, dx, True)
    dvdoubleprime_dy = first_order_derivative(v_double_prime, 2, dy, False)
    dvdoubleprime_dz = first_order_derivative(v_double_prime, 3, dz, True)

    dwdoubleprime_dx = first_order_derivative(w_double_prime, 1, dx, True)
    dwdoubleprime_dy = first_order_derivative(w_double_prime, 2, dy, False)
    dwdoubleprime_dz = first_order_derivative(w_double_prime, 3, dz, True)

    S_11 = dudoubleprime_dx + dudoubleprime_dx
    S_12 = dudoubleprime_dy + dvdoubleprime_dx
    S_13 = dudoubleprime_dz + dwdoubleprime_dx

    S_22 = dvdoubleprime_dy + dvdoubleprime_dy
    #S_21 =0     
    S_23 = dvdoubleprime_dz + dwdoubleprime_dy

    S_33 = dwdoubleprime_dz + dwdoubleprime_dz    
    #S_32 =0     
    #S_31 =0     

    
    '''
    Summation 
    '''
    t1 = mu
    t2 = (S_11 * S_11 + S_12 * S_12 + S_13 * S_13 +  \
          S_12 * S_12 + S_22 * S_22 + S_23 * S_23 +  \
          S_13 * S_13 + S_23 * S_23 + S_33 * S_33)
    #t2 = np.mean(t2, axis=(0, 2))
    epsilon = -0.5 * (np.mean(t1 * t2, axis=(0, 2)))

    q_parts     = case.comm.gather(epsilon, root=0)
    if(case.rank == 0):
        q_global     = np.concatenate(q_parts)
        
        #print(q_global.shape)
        plot(dy, q_global)

        #m = global_field_prime.mean(axis=(0, 2))
        #rms = np.sqrt((global_field_prime**2).mean(axis=(0,2)))  # rms vs y
        #rel = np.abs(m) / (rms + 1e-30)
        #print(rel.max())
    
    else:
        q_global     = None


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
    #plt.ylabel(r"$\overline{-u_i''\,\partial_i p'}_{xz}$")
    #plt.title("Pressure term profile (plane-averaged)")
    #plt.title("Turbulent Diffusion term profile (plane-averaged)")
    #plt.title("Advection term profile (plane-averaged)")
    plt.title("Turbulent Dissipiation term profile (plane-averaged)")
    plt.grid(True)
    plt.tight_layout()
   
    #plt.savefig("turbulent_diffusion_term.png", dpi=300)
    #plt.savefig("pressure_term.png", dpi=300)
    #plt.savefig("advection.png", dpi=300)
    plt.savefig("dissipiation.png", dpi=300)
    plt.close() 
    

def main():
    case, dx, dy, dz = large_case_update()

    #Pressure_Term 
    #pressure_term(case, dx, dy, dz)

    #Turbulent_Diffusion 
    #turbulent_diffusion(case, dx, dy, dz)
    
    #Advection
    #advection(case, dx, dy, dz)

    #Dissipiation
    dissipiation(case, dx, dy, dz)
    


if __name__ == "__main__":
    main()

