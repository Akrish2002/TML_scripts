# AI Assistance Declaration:
# Portions of this script were generated with the assistance of OpenAI ChatGPT (GPT-5.2).
# The generated code has only been lightly tested for basic functionality.
# Any errors, numerical inaccuracies, or unintended behavior must be rigorously
# investigated and validated by the user. The generated sections have not undergone
# manual review or verification beyond minimal execution checks.

# AI Logic Attribution:
# In a few instances, short code segments (one–two lines) were written by the author
# using algorithmic logic and reasoning suggested by ChatGPT (GPT-5.2).
# While the code implementation is entirely the author's, the underlying logic in
# these cases was AI-informed. Full responsibility for correctness and usage lies
# with the author.

from mpi4py import MPI
from FPCSLpy.case import LargeCase
import numpy as np
import os
import csv
import argparse
import re, pathlib
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import matplotlib as mpl


#AI-generated function (ChatGPT GPT-5.2).
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})


#AI-generated function (ChatGPT GPT-5.2).
colors = {
    "Advection"         : "#7B3294",    #Purple
    "Turb. diff."       : "#4DAF4A",    #Green
    "Production"        : "#E69F00",    #Orange
    "Dissipation"       : "#D62728",    #Red
    "Pres. without ST"  : "#1F78B4",    #Blue
    "Visc. diff."       : "#000000",    #Black
}


#AI-generated function (ChatGPT GPT-5.2).
def apply_paper_style(ax):
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, color="0.55")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("k")
    ax.tick_params(direction="out", length=4, width=1.0, colors="k")


class TKE_Budget:


    def grep_ctr(self, st, ctr_file=None):
        """ To grep all the required data from the CTR file
        
        Args:
            st (string) : The variable whose data is to be grep-ed
        
        Return:
            The variable's value
        """
        
        if ctr_file is None:
            ctr_file = self._ctr_file
        text = pathlib.Path(ctr_file).read_text()
        pat = re.compile(rf"\b{re.escape(st)}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
        m = pat.search(text)
        n   = float(m.group(1)) if m else None
    
        return n


    def __init__(self, path):
        """
        Grepping initial data and updating case
        """

        self._stackdirection = None
        self._time_step = None

        #Path
        self._path = path
        self._ctr_file = str(Path(path) / "incompressible_tml.ctr")

        #Flow variables
        self._mu_1 = self.grep_ctr("mu1")
        self._mu_2 = self.grep_ctr("mu2")

        self._case = LargeCase(self._path)
        
        #Grid
        self._nx_g = int(self.grep_ctr("nx"))
        self._ny_g = int(self.grep_ctr("ny"))
        self._nz_g = int(self.grep_ctr("nz"))

        #Domain
        self._xmax = self.grep_ctr("xmax")
        self._a    = int(self._xmax / np.pi)
        self._ymax = self.grep_ctr("ymax")
        self._b    = int(self._ymax / np.pi)
        self._zmax = self.grep_ctr("zmax")
        self._c    = int(self._zmax / np.pi)

        #Parallel
        self._nxsd = int(self.grep_ctr("nxsd"))
        self._nysd = int(self.grep_ctr("nysd"))
        self._nzsd = int(self.grep_ctr("nzsd"))

        #Update default parameters
        to_update_parameters = self._case.parameters
        to_update_parameters['grid']['x_max']                                       = self._a * np.pi
        to_update_parameters['grid']['y_max']                                       = self._b * np.pi
        to_update_parameters['grid']['z_max']                                       = self._c * np.pi
        to_update_parameters['grid']['nx']                                          = self._nx_g
        to_update_parameters['grid']['ny']                                          = self._ny_g
        to_update_parameters['grid']['nz']                                          = self._nz_g
        to_update_parameters['simulation_parameters']['parallel']['nxsd']           = self._nxsd
        to_update_parameters['simulation_parameters']['parallel']['nysd']           = self._nysd
        to_update_parameters['simulation_parameters']['parallel']['nzsd']           = self._nzsd
        to_update_parameters['simulation_parameters']['solvers']['incompressible']  = True

        self._dx = (self._case.parameters['grid']['x_max'] - self._case.parameters['grid']['x_min']) / self._case.parameters['grid']['nx']
        self._dy = (self._case.parameters['grid']['y_max'] - self._case.parameters['grid']['y_min']) / self._case.parameters['grid']['ny']
        self._dz = (self._case.parameters['grid']['z_max'] - self._case.parameters['grid']['z_min']) / self._case.parameters['grid']['nz']

        #Update and check parameters
        self._case.update_parameters(to_update_parameters)
        

    def grep_timestep(self):
        """ 
        Grepping time steps to calculate first step, step and last step
        
        Args:
        
        Return:
        
        """
        
        root = Path(self._path)
        nums = []
        
        for p in root.iterdir():
            m = re.fullmatch(r"time_step-(\d+)", p.name)
            if m:
                nums.append(int(m.group(1)))
        
        nums.sort()
        if nums:
            self._fs, self._step, self._ls = min(nums), nums[1] - nums[0], max(nums)
        

    def read_field_large(self, field, phi=False, alltimesteps=False):
        """
        asdf
        """
    
        self._case.distribute_block_list_axis_stack(self._stackdirection)
        filtered_block_list = self._case.filtered_rank_block_list
    
        a = int(self._nx_g / self._nxsd)
        b = int(self._ny_g / self._nysd)
        c = int(self._nz_g / self._nzsd)
        block_org         = np.empty((a, b, c, self._nzsd, self._nxsd))
        block_org_phi     = np.empty((a, b, c, self._nzsd, self._nxsd))
        block_avg         = np.empty((b))
        block_avg_phi     = np.empty((b))
    
        for i in range(self._nxsd):
            for k in range(self._nzsd): 
                nxr, nyr, nzr = self._case.get_nxrnyrnzr_from_nr(filtered_block_list[i * self._nzsd + k])
    
                block                    = self._case.read_block(self._time_step, nxr, nyr, nzr, to_read=[f'{field}', 'phi_1'], to_interpolate=True)
                block_org[..., k, i]     = block[f'{field}']
                block_org_phi[..., k, i] = block['phi_1']
                
                del(block)
    
        #Averaging
        block_avg     = np.mean(block_org,     axis=(0, 2, 3, 4)) 
        block_avg_phi = np.mean(block_org_phi, axis=(0, 2, 3, 4)) 
    
        #Logic informed by ChatGPT (GPT-5.2).
        #GPT helped me with the transpose idea, but there is a stack answer disucssing this which I must read.
        slab     = block_org.transpose(4, 0, 1, 3, 2)
        slab_phi = block_org_phi.transpose(4, 0, 1, 3, 2)
    
        block_org     = slab.reshape(a * self._nxsd, b, c * self._nzsd)
        block_org_phi = slab_phi.reshape(a * self._nxsd, b, c * self._nzsd)
    
        block_avg   = block_avg[None, :, None]
        block_prime = block_org - block_avg
    
        if(phi == True):
            return block_org, block_avg, block_prime, block_org_phi, block_avg_phi
        else:
            return block_org, block_avg, block_prime


    def first_order_derivative(self, field, direction, dx, periodic=False):
        """
        asdf
        """

        derivative = np.empty_like(field, dtype=float)

        length = field.shape[direction - 1]
        axis = direction - 1
        
        if(periodic):
          derivative = (np.roll(field, -1, axis) - np.roll(field, 1, axis)) / (2*dx)
        
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
        
          derivative[tuple(idx_center)] = (field[tuple(D_center_1)] - field[tuple(D_center_2)]) / (2*dx)
          derivative[tuple(idx_top_edge)] = (-3*field[tuple(D_top_edge_1)] + 4*field[tuple(D_top_edge_2)] - field[tuple(D_top_edge_3)]) / (2*dx)
          derivative[tuple(idx_bottom_edge)] = (3*field[tuple(D_bottom_edge_1)] - 4*field[tuple(D_bottom_edge_2)] + field[tuple(D_bottom_edge_3)]) / (2*dx)
        
        return derivative


    def common_terms(self):
        '''
        Pitfalls:
            Getting u', v', w' instead of u'', v'' and w'' and Reynolds avg terms instead of Favre avg, 
            but in this particular case I am getting reynolds avg, since it is equal density and incompressible?
        '''

        _, self._u_avg, self._u_double_prime, self._phi_1, _ = self.read_field_large('u', True)
        _, self._v_avg, self._v_double_prime                 = self.read_field_large('v')
        _, self._w_avg, self._w_double_prime                 = self.read_field_large('w')

        _, _, self._p_prime = self.read_field_large('p')

        self._mu = self._phi_1 * self._mu_1 + (1 - self._phi_1) * self._mu_2


    #1st Term
    def advection(self):
        '''
        asdf
        '''

        #Finding k
        #I have to make it as [None, :, None], so that I can multiply? Yea I need to
        #do that to broadcast them
        k = np.mean(0.5 * (self._u_double_prime * self._u_double_prime  + \
                           self._v_double_prime * self._v_double_prime  + \
                           self._w_double_prime * self._w_double_prime), axis=(0,2))
        k = k[None, :, None]
    
        #Constructing the terms inside each differential
        #t1 = u_avg * k
        t2 = self._v_avg * k
        #t3 = w_avg * k
    
        #Finding the differential
        #dt1_dx1 = first_order_derivative(t1, 1, dx, True) 
        dt2_dx2 = self.first_order_derivative(t2, 2, self._dy, False) 
        #dt3_dx3 = first_order_derivative(t3, 3, dz, True) 
    
        #Forming the summed term
        #q = d_dx1 + d_dx2 + d_dx3
        q                         = -dt2_dx2
        self._advection_parts     = self._case.comm.gather(q, root=0)
        
        if(self._case.rank == 0):
            self._advection_global = np.concatenate(self._advection_parts, axis=1)
            
            #This is just to convert it to (ny,) there is no change in value
            self._advection_global = np.mean(self._advection_global, axis=(0, 2))
    
        else:
            q_global     = None


    #2nd Term
    def pressure_diffusion(self):
        """
        Computing the pressure differential in the TKE budget
    
        """
    
        #Compute instantaneous differential term
        dpprime_dx1 = self.first_order_derivative(self._p_prime, 1, self._dx, True)
        dpprime_dx2 = self.first_order_derivative(self._p_prime, 2, self._dy, False)
        dpprime_dx3 = self.first_order_derivative(self._p_prime, 3, self._dz, True)
            
        #Form the summed term
        q = -(self._u_double_prime * dpprime_dx1 + self._v_double_prime * dpprime_dx2 + self._w_double_prime * dpprime_dx3)
    
        q_y = np.mean(q, axis=(0, 2))

        self._pressure_diffusion_parts     = self._case.comm.gather(q_y, root=0)
        
        if(self._case.rank == 0):
            self._pressure_diffusion_global     = np.concatenate(self._pressure_diffusion_parts)

        else:
            self._pressure_diffusion_global     = None
    
    
    #3rd Term
    def turbulent_diffusion(self):
        """
        asdf
        """
 
        vel_double_prime_square_sum = self._u_double_prime * self._u_double_prime  + \
                                      self._v_double_prime * self._v_double_prime  + \
                                      self._w_double_prime * self._w_double_prime 
     
        
        #Constructing the terms inside the differential
        #t1 = np.mean(u_double_prime * vel_double_prime_square_sum, axis=(0,2))
        t2 = 0.5 * (np.mean(self._v_double_prime * vel_double_prime_square_sum, axis=(0,2)))
        #t3 = np.mean(w_double_prime * vel_double_prime_square_sum, axis=(0,2))
    
        #If I do not do this I get an error in the derivative.
        #t1 = t1[None, :, None]
        t2 = t2[None, :, None]
        #t3 = t3[None, :, None]
    
        #Finding the differential
        #dt1prime_dx1 = first_order_derivative(t1, 1, dx, True) 
        dt2prime_dx2 = self.first_order_derivative(t2, 2, self._dy, False) 
        #dt3prime_dx3 = first_order_derivative(t3, 3, dz, True) 
    
        #Forming the summed term
        #q = dt1prime_dx1 - dt2prime_dx2 - dt3prime_dx3
        #q = -dt2prime_dx2
        q = dt2prime_dx2
        self._turbulent_diffusion_parts     = self._case.comm.gather(q, root=0)
        
        if(self._case.rank == 0):
            self._turbulent_diffusion_global     = np.concatenate(self._turbulent_diffusion_parts, axis=1)
            
            #This is just to convert it to (ny,) there is no change in value
            self._turbulent_diffusion_global = np.mean(self._turbulent_diffusion_global, axis=(0, 2))
        
        else:
            self._turbulent_diffusion_global     = None


    #4th Term
    def viscous_diffusion(self):
        '''
        Pitfalls:
            1. I convert favre averaging of the term into reynolds avg because 
               this specific case it is the same
        '''
       
        #asdf
        dudoubleprime_dx = self.first_order_derivative(self._u_double_prime, 1, self._dx, True)
        dudoubleprime_dy = self.first_order_derivative(self._u_double_prime, 2, self._dy, False)
        dudoubleprime_dz = self.first_order_derivative(self._u_double_prime, 3, self._dz, True)
    
        dvdoubleprime_dx = self.first_order_derivative(self._v_double_prime, 1, self._dx, True)
        dvdoubleprime_dy = self.first_order_derivative(self._v_double_prime, 2, self._dy, False)
        dvdoubleprime_dz = self.first_order_derivative(self._v_double_prime, 3, self._dz, True)
    
        dwdoubleprime_dx = self.first_order_derivative(self._w_double_prime, 1, self._dx, True)
        dwdoubleprime_dy = self.first_order_derivative(self._w_double_prime, 2, self._dy, False)
        dwdoubleprime_dz = self.first_order_derivative(self._w_double_prime, 3, self._dz, True)

        t1 = self._u_double_prime * dudoubleprime_dx + self._v_double_prime * dvdoubleprime_dx + self._w_double_prime * dwdoubleprime_dx + \
             self._u_double_prime * dudoubleprime_dx + self._v_double_prime * dudoubleprime_dy + self._w_double_prime * dudoubleprime_dz 

        t2 = self._u_double_prime * dudoubleprime_dy + self._v_double_prime * dvdoubleprime_dy + self._w_double_prime * dwdoubleprime_dy + \
             self._u_double_prime * dvdoubleprime_dx + self._v_double_prime * dvdoubleprime_dy + self._w_double_prime * dvdoubleprime_dz 

        t3 = self._u_double_prime * dudoubleprime_dz + self._v_double_prime * dvdoubleprime_dz + self._w_double_prime * dwdoubleprime_dz + \
             self._u_double_prime * dwdoubleprime_dx + self._v_double_prime * dwdoubleprime_dy + self._w_double_prime * dwdoubleprime_dz 

        T1 = np.mean(self._mu * t1, axis=(0, 2))
        T2 = np.mean(self._mu * t2, axis=(0, 2))
        T3 = np.mean(self._mu * t3, axis=(0, 2))

        '''
        Pitfalls:
            - Unsure if this should be a pitfall, nevertheless, when I take the 
            averaging as I did above in the x and z directions, the derivatives
            in the those directions will be zero, I stand to gain nothing by 
            computing them.

            - I also have to do the T2 = T2[None, :, None] so that the derivative
            can be computed
        '''
        #dT1_dx = self.first_order_derivative(T1, 1, self._dx, True)
        T2 = T2[None, :, None]
        dT2_dy = self.first_order_derivative(T2, 2, self._dy, False)
        #dT3_dz = self.first_order_derivative(T3, 3, self._dz, True)

        #Summing
        #viscous_diffusion = dT1_dx + dT2_dy + dT3_dz
        viscous_diffusion =  dT2_dy 
        self._viscous_diffusion_parts = self._case.comm.gather(viscous_diffusion, root=0)

        if(self._case.rank == 0):
            self._viscous_diffusion_global = np.concatenate(self._viscous_diffusion_parts, axis=1)
            
            '''
            Pitfalls:
                -I have to do this so that it reduces to a single column that can be plotted.
                Just converting from (1, ny, 1) --> (ny,)
            '''
            self._viscous_diffusion_global = np.mean(self._viscous_diffusion_global, axis=(0, 2))

        else:
            self._viscous_diffusion_global     = None
        

    #5th Term
    def dissipation(self):
        """
        asdf
        """

        #All terms
        dudoubleprime_dx = self.first_order_derivative(self._u_double_prime, 1, self._dx, True)
        dudoubleprime_dy = self.first_order_derivative(self._u_double_prime, 2, self._dy, False)
        dudoubleprime_dz = self.first_order_derivative(self._u_double_prime, 3, self._dz, True)
    
        dvdoubleprime_dx = self.first_order_derivative(self._v_double_prime, 1, self._dx, True)
        dvdoubleprime_dy = self.first_order_derivative(self._v_double_prime, 2, self._dy, False)
        dvdoubleprime_dz = self.first_order_derivative(self._v_double_prime, 3, self._dz, True)
    
        dwdoubleprime_dx = self.first_order_derivative(self._w_double_prime, 1, self._dx, True)
        dwdoubleprime_dy = self.first_order_derivative(self._w_double_prime, 2, self._dy, False)
        dwdoubleprime_dz = self.first_order_derivative(self._w_double_prime, 3, self._dz, True)
    
        S_11 = dudoubleprime_dx + dudoubleprime_dx
        S_12 = dudoubleprime_dy + dvdoubleprime_dx
        S_13 = dudoubleprime_dz + dwdoubleprime_dx
    
        S_22 = dvdoubleprime_dy + dvdoubleprime_dy
        #S_21 =0     
        S_23 = dvdoubleprime_dz + dwdoubleprime_dy
    
        S_33 = dwdoubleprime_dz + dwdoubleprime_dz    
        #S_32 =0     
        #S_31 =0     
    
        #Summation 
        t1 = self._mu
        t2 = (S_11 * S_11 + S_12 * S_12 + S_13 * S_13 +  \
              S_12 * S_12 + S_22 * S_22 + S_23 * S_23 +  \
              S_13 * S_13 + S_23 * S_23 + S_33 * S_33)
        epsilon = -0.5 * (np.mean(t1 * t2, axis=(0, 2)))
        self._dissipation_parts     = self._case.comm.gather(epsilon, root=0)

        if(self._case.rank == 0):
            self._dissipation_global     = np.concatenate(self._dissipation_parts)
            
        else:
            self._dissipation_global     = None


    #6th term
    def production(self):
        """
        asdf
        """

        #Naming to match favre averaging, I did not do this in others
        uu_f = np.mean(self._u_double_prime * self._u_double_prime, axis=(0, 2))
        uv_f = np.mean(self._u_double_prime * self._v_double_prime, axis=(0, 2))
        uw_f = np.mean(self._u_double_prime * self._w_double_prime, axis=(0, 2))

        vv_f = np.mean(self._v_double_prime * self._v_double_prime, axis=(0, 2))
        vw_f = np.mean(self._v_double_prime * self._w_double_prime, axis=(0, 2))

        ww_f = np.mean(self._w_double_prime * self._w_double_prime, axis=(0, 2))

        #Derivative terms
        du_dx = self.first_order_derivative(self._u_avg, 1, self._dx, True)
        du_dy = self.first_order_derivative(self._u_avg, 2, self._dy, False)
        du_dz = self.first_order_derivative(self._u_avg, 3, self._dz, True)

        dv_dx = self.first_order_derivative(self._v_avg, 1, self._dx, True)
        dv_dy = self.first_order_derivative(self._v_avg, 2, self._dy, False)
        dv_dz = self.first_order_derivative(self._v_avg, 3, self._dz, True)

        dw_dx = self.first_order_derivative(self._w_avg, 1, self._dx, True)
        dw_dy = self.first_order_derivative(self._w_avg, 2, self._dy, False)
        dw_dz = self.first_order_derivative(self._w_avg, 3, self._dz, True)

        '''
        Pitfalls:
            I do this just to reduce the size from (1, ny_part, 1) to (ny_part,)
            An elegant way would be to include it in the derivative or make everything
            use a singular format
        '''
        du_dx = np.mean(du_dx, axis=(0, 2))
        du_dy = np.mean(du_dy, axis=(0, 2))
        du_dz = np.mean(du_dz, axis=(0, 2))

        dv_dx = np.mean(dv_dx, axis=(0, 2))
        dv_dy = np.mean(dv_dy, axis=(0, 2))
        dv_dz = np.mean(dv_dz, axis=(0, 2))

        dw_dx = np.mean(dw_dx, axis=(0, 2))
        dw_dy = np.mean(dw_dy, axis=(0, 2))
        dw_dz = np.mean(dw_dz, axis=(0, 2))

        #Constructing the terms
        P_U = -(uu_f * du_dx + uv_f * dv_dx + uw_f * dw_dx)
        P_V = -(uv_f * du_dy + vv_f * dv_dy + vw_f * dw_dy)
        P_W = -(uw_f * du_dz + vw_f * dv_dz + ww_f * dw_dz)

        #Summing
        P = P_U + P_V + P_W
        self._production_parts     = self._case.comm.gather(P, root=0)

        if(self._case.rank == 0):
            self._production_global     = np.concatenate(self._production_parts)
            
        else:
            self._production_global     = None
            
        
    def sum_all_terms(self):
        if(self._case.rank == 0):
            self._sum_all_terms_global = self._advection_global + self._pressure_diffusion_global + \
                                         self._turbulent_diffusion_global + self._viscous_diffusion_global + \
                                         self._dissipation_global + self._production_global
        else:
            self._sum_all_terms_global = None


    #AI-generated function (ChatGPT GPT-5.2).
    def plot(
        self,
        title="TKE budget terms (plane-averaged)",
        fname="tke_budget.png",
        zoom_pi=False,
        pi_window=0.6,
        ylim=None,
        figsize=(7.2, 5.0),
        dpi=150,
    ):
        if self._case.rank != 0:
            return
    
        # y-grid (cell-centered)
        ny = self._ny_g
        y = (np.arange(ny) + 0.5) * (2*np.pi / ny)
    
        # optional zoom around y=pi
        if zoom_pi:
            x1 = np.pi - pi_window
            x2 = np.pi + pi_window
            mask = (y >= x1) & (y <= x2)
        else:
            mask = slice(None)
    
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
    
        # Order + legend labels (match your old names but “paper-like”)
        terms = [
            ("_advection_global", "Advection"),
            ("_turbulent_diffusion_global", "Turb. diff."),
            ("_production_global", "Production"),
            ("_dissipation_global", "Dissipation"),
            ("_pressure_diffusion_global", "Pres. diff."),
            ("_viscous_diffusion_global", "Visc. diff."),
            # ("_surface_tension_induced_diffusion_global", "Pres. (ST)"),  # if/when you add it
        ]
    
        # paper-like: single color, different dashes
        dash_cycle = ["-", ":", "--", "-.", (0, (5, 2)), (0, (3, 1, 1, 1))]
    
        plotted_any = False
        for i, (attr, lab) in enumerate(terms):
            val = getattr(self, attr, None)
            if val is None:
                continue
    
            arr = np.asarray(val).squeeze()
            if arr.ndim != 1 or arr.shape[0] != ny:
                raise ValueError(f"{attr} must be (ny,), got {np.asarray(val).shape}")
    
            ax.plot(
                y[mask],
                arr[mask],
                #color="k",                     # black lines like typical papers
                color=colors.get(lab, "k"),
                linestyle=dash_cycle[i % len(dash_cycle)],
                linewidth=1.2,
                label=lab,
            )
            plotted_any = True
    
        if not plotted_any:
            raise RuntimeError("No TKE term arrays were available on rank 0 for plotting.")
    
        ax.set_xlabel("y")
        ax.set_ylabel("TKE budget")
    
        if zoom_pi:
            ax.set_xlim(np.pi - pi_window, np.pi + pi_window)
    
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
    
        apply_paper_style(ax)
    
        # title inside plot area (paper-like)
        ax.text(
            0.50, 0.93, title,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=14, color="k",
        )
    
        ax.legend(loc="best", frameon=False)
    
        fig.tight_layout(pad=1.0)
        fig.savefig(fname, dpi=300)
        plt.close(fig)


def main():
    #TKE = TKE_Budget("/ccs/home/abhi/member-work/incompressible-tml/BaseCase/n512")
    #TKE = TKE_Budget("../")

    #TKE = TKE_Budget("../../n256/run3")
    #TKE._time_step = 5500
    #TKE._stackdirection = 1

    TKE = TKE_Budget("../")
    TKE._time_step = 12000
    TKE._stackdirection = 1

    #TKE = TKE_Budget("../../n1024/run6")
    #TKE._time_step = 108000
    #TKE._stackdirection = 1

    TKE.common_terms()
    TKE.advection()
    TKE.pressure_diffusion()
    TKE.turbulent_diffusion()
    TKE.viscous_diffusion()
    TKE.dissipation()
    TKE.production()

    #TKE.plot(
    #    fname="tke_budget_n256.png",
    #    title="M1(256³)",
    #    zoom_pi=True,
    #    pi_window=2.5,
    #)

    TKE.plot(
        fname="tke_budget_n512.png",
        title="M2(512³)",
        zoom_pi=True,
        pi_window=2.5,
    )

    #TKE.plot(
    #    fname="tke_budget_n1024.png",
    #    title="M3(1024³)",
    #    zoom_pi=True,
    #    pi_window=2.5,
    #)


if __name__ == "__main__":
    main()
