import numpy as np

def expand_array(a, repeat_size_x, repeat_size_y, repeat_size_z): 
    b = np.repeat(a, repeats=repeat_size_x, axis=0)
    b = np.repeat(b, repeats=repeat_size_y, axis=1)
    b = np.repeat(b, repeats=repeat_size_z, axis=2)
    return b

def params(Re, mu, delta, rho, We):
    u1 = (Re * mu)/delta #Re = (U1 * delta)/mu
    sigma = (rho * u1 * u1 * delta)/We #We = (rho * u1 * u1 * delta)/sigma
    print("U1=",u1, "Sigma=", sigma)
    return u1, sigma

if __name__ == '__main__':

    # Inputs
    time_step = 0

    from FPCSLpy.case import LargeCase
    read_case = LargeCase('./.')
    write_case = LargeCase('./TML_v3_n512_v3')

    # Update default parameters
    to_update_read_parameters = read_case.parameters  
    to_update_read_parameters['grid']['nx'] = 256
    to_update_read_parameters['grid']['ny'] = 256
    to_update_read_parameters['grid']['nz'] = 5
    to_update_read_parameters['simulation_parameters']['parallel']['nxsd'] = 4
    to_update_read_parameters['simulation_parameters']['parallel']['nysd'] = 4
    to_update_read_parameters['simulation_parameters']['parallel']['nzsd'] = 1 
    to_update_read_parameters['simulation_parameters']['solvers']['incompressible'] = True
    read_case.update_parameters(to_update_read_parameters)

    to_update_write_parameters = write_case.parameters  
    to_update_write_parameters['grid']['nx'] = 512 
    to_update_write_parameters['grid']['ny'] = 512 
    to_update_write_parameters['grid']['nz'] = 5
    to_update_write_parameters['simulation_parameters']['parallel']['nxsd'] = 8 
    to_update_write_parameters['simulation_parameters']['parallel']['nysd'] = 8 
    to_update_write_parameters['simulation_parameters']['parallel']['nzsd'] = 1 
    to_update_write_parameters['simulation_parameters']['solvers']['incompressible'] = True
    write_case.update_parameters(to_update_write_parameters)

    interpolation_factor_x, \
    interpolation_factor_y, \
    interpolation_factor_z = int(write_case.parameters['grid']['nx']
                                /read_case.parameters['grid']['nx']), \
                             int(write_case.parameters['grid']['ny'] 
                                /read_case.parameters['grid']['ny']), \
                             int(write_case.parameters['grid']['nz'] 
                                /read_case.parameters['grid']['nz'])
    block_factor_x, \
    block_factor_y, \
    block_factor_z = int(write_case.parameters['simulation_parameters']['parallel']['nxsd']
                        /read_case.parameters['simulation_parameters']['parallel']['nxsd']), \
                     int(write_case.parameters['simulation_parameters']['parallel']['nysd'] 
                        /read_case.parameters['simulation_parameters']['parallel']['nysd']), \
                     int(write_case.parameters['simulation_parameters']['parallel']['nzsd'] 
                        /read_case.parameters['simulation_parameters']['parallel']['nzsd'])

    sx, sy, sz = int(read_case.parameters['grid']['nx']/read_case.parameters['simulation_parameters']['parallel']['nxsd']), \
                 int(read_case.parameters['grid']['ny']/read_case.parameters['simulation_parameters']['parallel']['nysd']), \
                 int(read_case.parameters['grid']['nz']/read_case.parameters['simulation_parameters']['parallel']['nzsd'])

#    grid_parameters['x_min'], grid_parameters['x_max'] \
#    grid_parameters['y_min'], grid_parameters['y_max'] \
#    grid_parameters['z_min'], grid_parameters['z_max'] = write_case.parameters['grid']['x_min'], write_case.parameters['grid']['x_max'] \
                                                        # write_case.parameters['grid']['x_min'], write_case.parameters['grid']['x_max'] \
                                                        # write_case.parameters['grid']['x_min'], write_case.parameters['grid']['x_max']
    x0, xn, \
    y0, yn, \
    z0, zn = write_case.parameters['grid']['x_min'], write_case.parameters['grid']['x_max'], \
             write_case.parameters['grid']['y_min'], write_case.parameters['grid']['y_max'], \
             write_case.parameters['grid']['z_min'], write_case.parameters['grid']['z_max']

    nx, nxsd, \
    ny, nysd, \
    nz, nzsd =  write_case.parameters['grid']['nx'], write_case.parameters['simulation_parameters']['parallel']['nxsd'], \
                write_case.parameters['grid']['ny'], write_case.parameters['simulation_parameters']['parallel']['nysd'], \
                write_case.parameters['grid']['nz'], write_case.parameters['simulation_parameters']['parallel']['nzsd']
    
    xs = np.linspace(x0, xn, nx+1)
    xcs = 0.5*(xs[1:]+xs[:-1])
    ys = np.linspace(y0, yn, ny+1)
    ycs = 0.5*(ys[1:]+ys[:-1])
    zs = np.linspace(z0, zn, nz+1)
    zcs = 0.5*(zs[1:]+zs[:-1])

    #Parameters
    rho1, rho2 = 1., 1.
    mu1 , mu2  = 1.e-3, 5.e-2
#    U1  , U2   = 1., 0.
#    V1  , V2   = 0., 0.
    yloc = np.pi
    epsilon = 0.51*((xn-x0)/nx) 
    delta = 2.*np.pi/100.
    Re = 200.
    We = 20.
    U1, sigma = params(Re, mu1, delta, rho1, We)

    rank_block_list = read_case.block_list
    for small_block_nr in rank_block_list:
        #print(f'Interpolating of block {small_block_nr} within {to_read[0]} and {to_read[-1]}')
        nxr, nyr, nzr = read_case.get_nxrnyrnzr_from_nr(small_block_nr) 
        small_block   = read_case.read_block(time_step, nxr, nyr, nzr, to_interpolate=True)
        big_block     = write_case.create_default_block()

        nxnp = int(nx/nxsd)
        nynp = int(ny/nysd)
        nznp = int(nz/nzsd)
        
        for i in range(block_factor_x):
            for j in range(block_factor_y):
                for k in range(block_factor_z):

                    # Interpolate uc, vc, wc, p
                    big_block['uc'] = expand_array(small_block['uc'][int(sx/block_factor_x)*i:int(sx/block_factor_x)*(i+1),int(sy/block_factor_y)*j:int(sy/block_factor_y)*(j+1),int(sz/block_factor_z)*k:int(sz/block_factor_z)*(k+1)], 
                                                   interpolation_factor_x, interpolation_factor_y, interpolation_factor_z)
                    big_block['vc'] = expand_array(small_block['vc'][int(sx/block_factor_x)*i:int(sx/block_factor_x)*(i+1),int(sy/block_factor_y)*j:int(sy/block_factor_y)*(j+1),int(sz/block_factor_z)*k:int(sz/block_factor_z)*(k+1)], 
                                                   interpolation_factor_x, interpolation_factor_y, interpolation_factor_z)
                    big_block['wc'] = expand_array(small_block['wc'][int(sx/block_factor_x)*i:int(sx/block_factor_x)*(i+1),int(sy/block_factor_y)*j:int(sy/block_factor_y)*(j+1),int(sz/block_factor_z)*k:int(sz/block_factor_z)*(k+1)], 
                                                   interpolation_factor_x, interpolation_factor_y, interpolation_factor_z)
                    big_block['p']  = expand_array(small_block['p'][int(sx/block_factor_x)*i:int(sx/block_factor_x)*(i+1),int(sy/block_factor_y)*j:int(sy/block_factor_y)*(j+1),int(sz/block_factor_z)*k:int(sz/block_factor_z)*(k+1)], 
                                                   interpolation_factor_x, interpolation_factor_y, interpolation_factor_z)
                    # local grid from big case
                    # need to get the nxr/nyr/nzr of big case
                    l_nxr, l_nyr, l_nzr = block_factor_x*nxr+i, block_factor_y*nyr+j, block_factor_z*nzr+k
                    l_xcs = xcs[l_nxr*nxnp:(l_nxr+1)*nxnp]
                    l_ycs = ycs[l_nyr*nynp:(l_nyr+1)*nynp]
                    l_zcs = zcs[l_nzr*nznp:(l_nzr+1)*nznp]

                    l_grid_xcs, l_grid_ycs, l_grid_zcs = np.meshgrid(l_xcs, l_ycs, l_zcs, indexing='ij')

                    phi1 = 0.5*(1.+np.tanh((l_grid_ycs-yloc)/(2.*epsilon)))
                    big_block['phi_1']  = phi1 
                    big_block['phi_2']  =  1 - phi1
                    big_block['rho']  = phi1*rho1 + (1. - phi1)*rho2
                    big_block['mu']   = phi1*mu1  + (1. - phi1)*mu2
 
                    write_case.write_block(0, block_factor_x*nxr+i, block_factor_y*nyr+j, block_factor_z*nzr+k, big_block, to_interpolate=True)
        del(small_block, big_block)





