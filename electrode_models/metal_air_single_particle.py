"""
    metal_air_single_particle.py

    Class file for metal air electrode methods
"""

import cantera as ct
from math import tanh
import numpy as np

class electrode(): 
    """
    Create an electrode object representing the metal air electrode.
    """

    def __init__(self, input_file, inputs, sep_inputs, counter_inputs,    
        electrode_name, params, offset):
        """
        Initialize the model.
        """ 
   
        # Import relevant Cantera objects.
        self.gas_obj = ct.Solution(input_file, inputs['gas-phase'])
        self.elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])
        self.gas_elyte_obj = ct.Interface(input_file, inputs['elyte-iphase'], [self.gas_obj, self.elyte_obj])
        self.host_obj = ct.Solution(input_file, inputs['host-phase'])
        self.product_obj = ct.Solution(input_file, inputs['product-phase'])
        self.surf_obj = ct.Interface(input_file, inputs['surf-iphase'], 
            [self.product_obj, self.elyte_obj, self.host_obj])

        # Anode or cathode? Positive external current delivers positive charge 
        # to the anode, and removes positive charge from the cathode.
        self.name = electrode_name
        if self.name=='anode':
            self.i_ext_flag = -1
        elif self.name=='cathode':
            self.i_ext_flag = 1
        else:
            raise ValueError("Electrode must be an anode or a cathode.")

        # Store the species index of the Li ion in the Cantera object for the 
        # electrolyte phase:
        self.index_Li = self.elyte_obj.species_index(inputs['mobile-ion'])

        # Electrode thickness and inverse thickness:
        self.dy = inputs['thickness']
        self.dyInv = 1/self.dy
        self.N_y = inputs['n-points'] # change to n-points

        # Phase volume fractions
        self.eps_host = inputs['eps_host']
        self.eps_product_init =inputs['eps_product']
        self.eps_elyte_init = 1 - self.eps_host - self.eps_product_init

        # This calculation assumes spherical particles of a single radius, with 
        # no overlap.
        # Electrode-electrolyte interface area, per unit geometric area.
        self.geometry = inputs['geometry']
        self.r_host = inputs['r_host']
        self.th_oxide = inputs['th_oxide']
        self.particle_an= inputs['particle_an']
        self.particle_int = inputs['particle_int']
        self.r_oxide = inputs['r_oxide']
        self.r_oxint = inputs['r_ox_int']
        self.V_host = 4./3. * np.pi * (self.r_host / 2)**3  # carbon or host volume [m3]
        self.A_host = 4. * np.pi * (self.r_host / 2)**2    # carbon or host surface area [m2]
        self.A_init = self.eps_host * self.A_host / self.V_host  # m2 of interface / m3 of total volume [m-1]
        self.A_oxide = np.pi* self.r_oxide**2/4.   # oxide area
        self.V_oxide = 2./3. * np.pi* (self.r_oxide/2.)**2 * self.th_oxide #oxide volume

        # For some models, the elyte thickness is different from that of the 
        # electrode, so we specify is separately:
        self.dy_elyte = self.dy
        self.dy_elyte_node = self.dy_elyte/self.N_y

        # Inverse double layer capacitance, per unit interfacial area.
        self.C_dl_Inv = 1/inputs['C_dl']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        self.elyte_microstructure = self.eps_elyte_init**1.5 # where would we use this?
        
        # SV_offset specifies the index of the first SV variable for the 
        # electode (zero for anode, n_vars_anode + n_vars_sep for the cathode)
        self.SV_offset = offset

        # Determine the electrode capacity (Ah/m2)

        # Max voume concentration of the product species (assuming all 
        # electrolyte has been replaced by oxide)
        stored_species = inputs['stored-species']
        v_molar_prod = \
            self.product_obj[stored_species['name']].partial_molar_volumes[0]

        self.capacity = (stored_species['charge']*ct.faraday
                * self.eps_elyte_init * inputs['thickness']
                / (3600 * v_molar_prod))
                 
        # Minimum volume fraction for the product phase, below which product 
        # phase consumption reaction shut off:
        self.product_phase_min = inputs['product-phase-min']
        # Number of state variables: electrode potential, electrolyte composition, oxide volume fraction 
        self.n_vars = 3 + self.elyte_obj.n_species
        self.n_vars_tot = self.N_y*self.n_vars

        # This model produces zero plots, but might someday.
        self.n_plots = 2

        # Store any extra species to be ploted
        self.plot_species = []
        [self.plot_species.append(sp['name']) for sp in inputs['plot-species']]

        # Set Cantera object state:
        self.host_obj.TP = params['T'], params['P']
        self.elyte_obj.TP = params['T'], params['P']
        self.surf_obj.TP = params['T'], params['P']
        #self.conductor_obj.TP = params['T'], params['P']

        # Set up pointers to specific variables in the solution vector:
        self.SVptr = {}
        self.SVptr['phi_ed'] = np.arange(0, self.n_vars_tot, self.n_vars)
        self.SVptr['phi_dl'] = np.arange(1, self.n_vars_tot, self.n_vars)
        # if self.particle_an == 'no': 
        self.SVptr['eps_product'] = np.arange(2, self.n_vars_tot, self.n_vars)
        # elif self.particle_an == 'yes':
        #     self.SVptr['radius'] = np.arange(2, self.n_vars_tot, self.n_vars)
        self.SVptr['C_k_elyte'] = np.ndarray(shape=(self.N_y, 
            self.elyte_obj.n_species), dtype='int')       
        for i in range(self.N_y):
            self.SVptr['C_k_elyte'][i,:] = range(3 + i*self.n_vars, 
                3 + i*self.n_vars + self.elyte_obj.n_species)

        # A pointer to where the SV variables for this electrode are, within the 
        # overall solution vector for the entire problem:
        self.SVptr['electrode'] = np.arange(offset, offset+self.n_vars_tot)

        # Save the indices of any algebraic variables:
        self.algvars = offset + self.SVptr['phi_ed'][:]

    def initialize(self, inputs, sep_inputs):

        # Initialize the solution vector for the electrode domain:
        SV = np.zeros(self.n_vars_tot)

        # Load intial state variables: Change it later
        SV[self.SVptr['phi_ed']] = inputs['phi_0']
        SV[self.SVptr['phi_dl']] = sep_inputs['phi_0'] - inputs['phi_0']
        # if self.particle_an == 'no':
        SV[self.SVptr['eps_product']] = self.eps_product_init
        # if self.particle_an == 'yes':
        #     SV[self.SVptr['radius']] = self.r_oxint
        SV[self.SVptr['C_k_elyte']] = self.elyte_obj.concentrations
        
        return SV

    def residual(self, t, SV, SVdot, sep, counter, params):
        """
        Define the residual for the state of the metal air electrode.

        This is an array of differential and algebraic governing equations, one for each state variable in the anode (anode plus a thin layer of electrolyte + separator).

        1. The electric potential in the electrode phase is an algebraic variable.
            In the anode, phi = 0 is the reference potential for the system.
            In the cathode, the electric potential must be such that the ionic current is spatially invariant (i.e. it is constant and equal to the external applied current, for galvanostatic simulations).  

            The residual corresponding to these variables (suppose an index 'j') are of the form:
                resid[j]  = (epression equaling zero)

        2. All other variables are governed by differential equations.
        
            We have a means to calculate dSV[j]/dt for a state variable SV[j] (state variable with index j).  
        
            The residuals corresponding to these variables will have the form:
                resid[j] = SVdot[j] - (expression equalling dSV/dt)

        Inputs:
            - SV: the solution vector representing the state of the entire battery domain.
            - SVdot: the time derivative of each state variable: dSV/dt
            - electrode: the object representing the current electrode
            - sep: the object representing the separator
            - counter: the object representing the electrode counter to the current electrode
            - params: dict of battery simulation parameters.
        """
        # Initialize the residual:
        resid = np.zeros((self.n_vars_tot,))

        # Save local copies of the solution vectors, pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]
        SVdot_loc = SVdot[SVptr['electrode']]
        dSVdt = np.zeros_like(SVdot_loc) # maybe change this
        
        # Read out properties:
        phi_ed = SV_loc[SVptr['phi_ed']]
        phi_elyte = phi_ed + SV_loc[SVptr['phi_dl']]
        # print('phi_elyte = ', phi_elyte)
        c_k_elyte = SV_loc[SVptr['C_k_elyte']]
        # if self.particle_an == 'no':
        eps_product = SV_loc[SVptr['eps_product']]
        # elif self.particle_an == 'yes':
        #     radius = SV_loc[SVptr['radius']]
        #     eps_product = self.eps_product_init + self.particle_int*2./3.*radius**3*np.pi/self.dyInv
           
        
        # Set Cantera object properties:
        self.host_obj.electric_potential = phi_ed
        self.elyte_obj.electric_potential = phi_elyte
        self.elyte_obj.X = c_k_elyte

        # Set microstructure multiplier for effective diffusivities
        eps_elyte = 1. - eps_product - self.eps_host
        self.elyte_microstructure = eps_elyte**1.5

        # Read electrolyte fluxes at the separator boundary:
        N_k_sep, i_io_sep = sep.electrode_boundary_flux(SV, self, params['T']) 
        
        N_k_sep, i_io_sep = sep.electrode_boundary_flux(SV, self, params['T']) 
        j_index = np.arange(0,self.N_y+1, 1)

        if self.name=='anode':
            # The electric potential of the anode = 0 V.
            resid[[SVptr['phi_ed'][0]]] = phi_ed

        elif self.name=='cathode':
            # For the cathode, the potential of the cathode must be such that 
            # the electrolyte electric potential (calculated as phi_ca + 
            # dphi_dl) produces the correct ionic current between the separator # and cathode:
            if params['boundary'] == 'current':
                resid[SVptr['phi_ed']] = i_io_sep - params['i_ext']
            elif params['boundary'] == 'potential':                  
                resid[SVptr['phi_ed']] = (SV_loc[SVptr['phi_ed']] 
                    - params['potential']) 

        # Calculate available surface area (m2 interface per m3 electrode):
        if self.geometry == 'rectangle':
            A_avail = self.A_init - eps_product/self.th_oxide
        elif self.geometry == 'hemisphere':
            A_avail = self.A_init - 3./2.*eps_product/self.r_oxide
        elif self.geometry == 'toroid':
            A_avail = self.A_init - 2*eps_product/self.r_oxide/np.pi
        
        # Convert to m2 interface per m2 geometric area:
        A_surf_ratio = A_avail*self.dy

        # Molar production rate of electrolyte species at the electrolyte-air 
        # interface (kmol / m2 of interface / s)
        sdot_elyte_air = \
            self.gas_elyte_obj.get_net_production_rates(self.elyte_obj)

        # Multiplier to scale phase destruction rates.  As eps_product drops 
        # below the user-specified minimum, any reactions that consume the 
        # phase have their rates quickly go to zero:
        mult = tanh(eps_product / self.product_phase_min)

        # Chemical production rate of the product phase: (mol/m2 interface/s)
        sdot_product = (self.surf_obj.get_creation_rates(self.product_obj)
            - mult * self.surf_obj.get_destruction_rates(self.product_obj))
        # if self.particle_an == 'no':
        resid[SVptr['eps_product']] = (SVdot_loc[SVptr['eps_product']] 
            - A_avail * np.dot(sdot_product, self.product_obj.partial_molar_volumes))
        # elif self.particle_an == 'yes':
        #     resid[SVptr['radius']] = (3*np.dot(sdot_product, self.product_obj.partial_molar_volumes)/(self.particle_int*np.pi*2.))**(1./3.)
        # Rate of change of the product phase volume fraction:
        

        # Production rate of the electron (moles / m2 interface / s)
        sdot_electron = (mult * self.surf_obj.get_creation_rates(self.host_obj)
            - self.surf_obj.get_destruction_rates(self.host_obj))

        # Positive Faradaic current corresponds to positive charge created in 
        # the electrode:
        i_Far = -(ct.faraday * sdot_electron)

        # Double layer current has the same sign as i_Far
        i_dl = self.i_ext_flag*(i_io_sep)/A_surf_ratio - i_Far
        resid[SVptr['phi_dl']] = SVdot_loc[SVptr['phi_dl']] - i_dl*self.C_dl_Inv
        #change in concentration
        sdot_elyte_host = (mult*self.surf_obj.get_creation_rates(self.elyte_obj)
            - self.surf_obj.get_destruction_rates(self.elyte_obj))
        sdot_elyte_host[self.index_Li] -= i_dl / ct.faraday 
        
        # print(sdot_product, sdot_elyte_host)
        resid[SVptr['C_k_elyte']] = (SVdot_loc[SVptr['C_k_elyte']] 
<<<<<<< HEAD
            - (N_k_sep + sdot_elyte_air + sdot_elyte_host * A_surf_ratio) 
            * self.dyInv)
=======
            - (self.i_ext_flag*N_k_sep + sdot_elyte_air 
            + sdot_elyte_host * A_surf_ratio) * self.dyInv)
            
>>>>>>> 2de144626 (Metal-air with HNG kinetics.)
        return resid
        
    def voltage_lim(self, SV, val):
        """
        Check to see if the voltage limits have been exceeded.
        """
        # Save local copies of the solution vector and pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]
        
        # Calculate the current voltage, relative to the limit.  The simulation 
        # looks for instances where this value changes sign (i.e. where it 
        # crosses zero)    
        voltage_eval = SV_loc[SVptr['phi_ed']] - val
        
        return voltage_eval

    def adjust_separator(self, sep):
        """ 
        Sometimes, an electrode object requires adjustments to the separator object.  This is not the case, for the SPM.
        """

        # Return the separator class object, unaltered:
        return sep

    def output(self, axs, solution, ax_offset):
        """Plot the intercalation fraction vs. time"""
        eps_product_ptr = (2 + self.SV_offset + self.SVptr['eps_product'][0])
        
        axs[ax_offset].plot(solution[0,:]/3600, solution[eps_product_ptr, :])
        axs[ax_offset].set_ylabel(self.name+' product \n volume fraction')

        for name in self.plot_species:
            species_ptr = self.elyte_obj.species_index(name)
            C_k_elyte_ptr = (2 + self.SV_offset 
                + self.SVptr['C_k_elyte'][0, species_ptr])
            axs[ax_offset+1].plot(solution[0,:]/3600, 
                1000*solution[C_k_elyte_ptr,:])

        axs[ax_offset+1].legend(self.plot_species)
        axs[ax_offset+1].set_ylabel('Elyte Species Conc. \n (mol m$^{-3}$)')
        return axs

    #What is the small here? Is it to avoid zeros?
    #def read_state():
    

    def mass_fluxes(self, SV_loc, SVptr, params, j):
        
        #FIX: Actual diffusion  Question: best way to pull this 
        D_k = {}
        D_k['Li+[elyt])'] = 4e-11          # bulk diff coeff Li+ in elyte (m2/s)
        D_k['TFSI-[elyt]'] = 4e-13         # bulk diff coeff PF6- in elyte (m2/s)
        D_k['O2(e)'] = 7e-12           # bulk diff coeff O2 in elyte (m2/s)
        D_k['C10H22O5[elyt]'] = 1.           # EC diffusion is fast
        D_k['Li2O2[elyt]'] = 1.           # EC diffusion is fast

        D_k_temp = np.array([1.11e-10, 6.98e-11, 8.79e-11, 4.26e-11, 2e-13])
        phi_ed = SV_loc[SVptr['phi_ed'][j]]
        phi_elyte = phi_ed + SV_loc[SVptr['phi_dl'][j]]

        phi_ed_next = SV_loc[SVptr['phi_ed'][j]]
        phi_elyte_next = phi_ed_next + SV_loc[SVptr['phi_dl'][j-1]]
        #convert from mol/m3 to kg/m3
        rho_k = SV_loc[SVptr['C_k_elyte'][j]] 
        rho_k_next = SV_loc[SVptr['C_k_elyte'][j-1]]
       
        eps_oxide = SV_loc[SVptr['eps_oxide'][j]]
        eps_elyte = 1 - eps_oxide - self.eps_host
        eps_oxide_next =  SV_loc[SVptr['eps_oxide'][j-1]]
        eps_elyte_next = 1 - eps_oxide_next - self.eps_host
       # Take averages to find interface values.  Eventually this should be 
       # weighted by the volume dimensions:
        rho_k_avg = (rho_k + rho_k_next)/2.
        eps_avg = (eps_elyte+eps_elyte_next)/2.

        D_k_elyte = D_k_temp *eps_avg
        D_k_mig = D_k_elyte*self.elyte_obj.charges*ct.faraday/(ct.gas_constant*params['T'])*rho_k_avg#Question: easiest way to access this from yaml file
        #Question: is that right for cantera charages?

        #Question: is this a rate? or is this a concentration?  Re: Amy's code
        
        N_k = (D_k_elyte*(rho_k/eps_elyte- rho_k_next/eps_elyte_next) + D_k_mig*(phi_elyte - phi_elyte_next))*self.dyInv

        i_io = np.dot(N_k, self.elyte_obj.charges)*ct.faraday

        return  phi_elyte, eps_oxide, eps_elyte, N_k, i_io

#Official Soundtrack:
    #Cursive - Happy Hollow
    #Japancakes - If I Could See Dallas
    #Jimmy Eat World - Chase the Light + Invented
    #Lay - Lit
    #George Ezra - Staying at Tamara's