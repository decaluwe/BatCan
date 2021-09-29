"""
    nonintercalating_interphase.py

    Class file for nonintecalating electrode with sei/cei growth. Can also likely be used for anode-free battery simulations.
"""

import cantera as ct
import numpy as np

class electrode():
    """
    Create an electrode object representing the nonintercalating electrode
    """
    def __init__(self, input_file, inputs, sep_inputs, counter_inputs,
        electrode_name, params, offset):
        """
        Initialize the electrode object.
        """

        # Import relevant Cantera objects.
        # Nonintercalating electrode, electrolyte, interphase, and interphase 
        # electron conductor objects:
        self.bulk_obj = ct.Solution(input_file, inputs['bulk-phase'])
        self.elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])
        self.interphase_obj = ct.Solution(input_file, inputs['interphase'])
        self.conductor_obj = ct.Solution(input_file, inputs['conductor-phase'])

        # Interface objects:
        self.bulk_elyte = ct.Interface(input_file, 
            inputs['bulk-elyte-interface'], 
            [self.bulk_obj, self.elyte_obj, self.interphase_obj])
        self.bulk_interphase = ct.Interface(input_file,
            inputs['bulk-interphase-interface'], 
            [self.bulk_obj, self.interphase_obj, self.conductor_obj])
        self.interphase_elyte = ct.Interface(input_file, 
            inputs['interphase-elyte-interface'], 
            [self.interphase_obj, self.elyte_obj, self.conductor_obj])

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

        # Inverse of the double layer capacitance, per unit interface area:
        self.C_dl_Elyte_Inv = 1 / inputs['C_dl_elyte']
        self.C_dl_bulk_Inv = 1 / inputs['C_dl_bulk']
        
        # Electrolyte volume fraction in the separator:
        self.eps_elyte = sep_inputs['eps_electrolyte'] * (1. - inputs['eps_0'])

        # Maximum allowed interphase volume fraction:
        self.eps_max = inputs['eps_max']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        self.elyte_microstructure = self.eps_elyte**1.5
        
        # SV_offset specifies the index of the first SV variable for the 
        # electode (zero for anode, n_vars_anode + n_vars_sep for the cathode)
        self.SV_offset = offset
        
        # The interphase object is not capacity-limiting, in and of itself.  
        # Rather the total amount of Li in the system is the limit. This is 
        # done in a separate routine, at a later time. Provide a large 
        # placeholder number here, so that it will not be the minimum, when 
        # evaluated later:
        self.capacity = 1e21

        # Domain information:
        # Thickness of separator node considered as part of the anode domain.  
        # This is "subtracted" off from the total separator thickness.
        self.dy_elyte = inputs['dy_elyte']
        # Size of an interphase 'particle'
        self.d_interphase = inputs['d_interphase']
        # Number of discretized volumes:
        self.n_points = int(self.dy_elyte / self.d_interphase)
            
        # Mumber of state variables: interphase electric potential, electrolyte 
        # electric potential, interphase species concentration (n_species), and 
        # electrolyte composition (n_species)
        self.n_vars_node = (2 + self.interphase_obj.n_species 
            + self.elyte_obj.n_species)

        # Tottal number of variables (extra 1 is for the electrode electric 
        # potential):
        self.n_vars = 1 + self.n_vars_node * self.n_points
        # Number of plots produced (temporarily zero):
        self.n_plots = 0
        
        # Set the initial Cantera object states:
        # Electric Potential:
        self.bulk_obj.electric_potential = inputs['phi_0']
        self.conductor_obj.electric_potential = inputs['phi_0']
        self.interphase_obj.electric_potential = inputs['phi_0']
        self.elyte_obj.electric_potential = sep_inputs['phi_0']
        # If the user provided an initial composition, use that, here:
        if 'X_0' in inputs:
            self.interphase_obj.TPX = params['T'], params['P'], inputs['X_0']
        else:
            self.interphase_obj.TP = params['T'], params['P']

        self.bulk_obj.TP = params['T'], params['P']
        self.elyte_obj.TP = params['T'], params['P']
        self.conductor_obj.TP = params['T'], params['P']

        self.bulk_elyte.TP = params['T'], params['P']
        self.bulk_interphase.TP = params['T'], params['P']
        self.interphase_elyte.TP = params['T'], params['P']
        
    def initialize(self, inputs, sep_inputs):
        
        # Initialize the solution vector for the electrode domain:
        SV = np.zeros([self.n_vars])

        # Set up pointers to specific variables in the solution vector:
        self.SVptr = {}

        # A pointer to where the SV variables for this electrode are, within 
        # the overall solution vector for the entire problem:
        self.SVptr['electrode'] = np.arange(self.SV_offset, 
            self.SV_offset+self.n_vars)
        
        """ Define pointers to state variables and load initial state variables:"""
        # Electric potential of the non-intercalating electrode:
        self.SVptr['phi_ed'] = np.array([0])
        SV[self.SVptr['phi_ed']] = inputs['phi_0']

        # Electric potential of the electrolyte phase in the domain:
        self.SVptr['phi_elyte'] = np.arange(
            1, self.n_vars, self.n_vars_node, dtype='int')
        SV[self.SVptr['phi_elyte']] = sep_inputs['phi_0']

        # Double layer potential at interphase-elyte boundary:
        self.SVptr['phi_dl'] = np.arange(2, self.n_vars, self.n_vars_node, 
            dtype='int')
        SV[self.SVptr['phi_dl']] = sep_inputs['phi_0'] - inputs['phi_0'] 
        
        # Molar concentration of interphase species (kmol / m3 of total volume):
        self.SVptr['C_k_interphase'] = np.ndarray(
            shape = (self.n_points, self.interphase_obj.n_species), dtype='int')

        # Molar concentration of electrolyte species (kmol / m3 of electrolyte 
        # volume):
        self.SVptr['C_k_elyte'] = np.ndarray(
            shape = (self.n_points, self.elyte_obj.n_species), dtype='int')
        for j in range(self.n_points):
            self.SVptr['C_k_interphase'][j,:] = np.arange(
                3 + j*self.n_vars_node, 
                3 + j*self.n_vars_node + self.interphase_obj.n_species, 
                dtype = int)
            self.SVptr['C_k_elyte'][j,:] = np.arange(
                3 + j*self.n_vars_node+ self.interphase_obj.n_species, 
                3 + j*self.n_vars_node + self.interphase_obj.n_species 
                + self.elyte_obj.n_species, dtype = int)

        SV[self.SVptr['C_k_interphase']] = \
            self.interphase_obj.concentrations * inputs['eps_0']
        SV[self.SVptr['C_k_elyte']] = self.elyte_obj.concentrations

        # Save the SV indices of any algebraic variables:
        self.algvars = np.hstack((self.SV_offset + self.SVptr['phi_ed'][:]))#, 
            # self.SV_offset + self.SVptr['phi_interphase'][:]))

        return SV

    def residual(self, t, SV, SVdot, sep, counter, params):
        """
        Define the residual for the state of the nonintercalating electrode + interphase.

        This is an array of differential and algebraic governing equations, one for each state variable in the domain (nonintercalating electrode electric potential + layer of electroltye / separator).

        1. The electric potential is an algebraic variable.
            In the anode, phi = 0 is the reference potential for the system.
            In the cathode, the electric potential must be such that the ionic current is spatially in_variant (i.e. it is constant and equal to the external applied current, for galvanostatic simulations).  

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
    
        # Initialize the residual array:
        resid = np.zeros((self.n_vars,))

        # Save local copies of the solution vectors, pointers for this electrode:
        SVptr = self.SVptr
        SV_loc = SV[SVptr['electrode']]
        SVdot_loc = SVdot[SVptr['electrode']]

        resid = SVdot_loc

        i_el = np.zeros(self.n_points + 1,)
        i_io = np.zeros(self.n_points + 1,)
        i_Far = np.zeros(self.n_points,)        

        # Flux of electrolyte species between the separator and the electrolyte 
        # in the current electrode domain:
        N_k_sep, i_io_out = sep.electrode_boundary_flux(SV, self, params['T'])

        # Read local SEI volume fraction:
        C_k_interphase = SV_loc[SVptr['C_k_interphase'][:]]
        eps_interphase = np.dot(self.interphase_obj.partial_molar_volumes,  
            C_k_interphase.T)
        
        # Temporarily, only one node:
        j = 0
        phi_ed = SV_loc[SVptr['phi_ed'][j]]
        phi_elyte = SV_loc[SVptr['phi_elyte'][j]]
        phi_interphase = phi_elyte - SV_loc[SVptr['phi_dl'][j]]

        self.bulk_obj.electric_potential = phi_ed
        self.interphase_obj.electric_potential = phi_interphase
        self.conductor_obj.electric_potential = phi_interphase
        self.elyte_obj.electric_potential = phi_elyte

        # The current in the interphase entering this volume is that produced by
        #   charge-transfer reactions at the electrode-interphase interface:
        i_el_in = (eps_interphase 
            * self.bulk_interphase.get_net_production_rates(self.bulk_obj)
            * ct.faraday)
        
        print(phi_interphase)
        # print(i_el_in)

        i_Far = 0
        i_el_out = 0
        i_dl = i_Far - i_el_in + i_el_out
        resid[SVptr['phi_dl']] = (SVdot_loc[SVptr['phi_dl']] 
            - i_dl * self.C_dl_bulk_Inv)
        # print(i_dl * self.C_dl_bulk_Inv)

        # Electrode electric potential
        if self.name=='anode':
            # For the anode, the electric potential is an algebraic variable, 
            # always equal to zero:
            resid[SVptr['phi_ed']] = SV_loc[SVptr['phi_ed']]

        elif self.name=='cathode':
            # For the cathode, the potential of the cathode must be such that 
            # the electrolyte electric potential (calculated as phi_ca + 
            # dphi_dl) produces the correct ionic current between the separator # and cathode:
            if params['boundary'] == 'current':
                resid[SVptr['phi_ed']] = i_io_out - params['i_ext']
            elif params['boundary'] == 'potential':
                # Potential at time t:
                phi = np.interp(t, params['times'], params['potentials'])

                # Cell potential must equal phi:
                resid[SVptr['phi_ed']] = SV_loc[SVptr['phi_ed']] - phi

                # TEMPORARY:
                # resid[SVptr['phi_interphase']] = \
                #     SV_loc[SVptr['phi_interphase']] - phi


        
        return resid

    def adjust_separator(self, sep):
        """
        The electrode domain considers the electrode object plus a thin layer of the separator, adjacent to the self. The interphase grows into this domain. We subtract this thickness from the total separator thickness, so that we do not inadvertently increase the total transport resistance through the separator.
        """
        # New separator thickness:
        thickness = sep.dy * sep.n_points - self.dy_elyte        

        # Reduce the number of points in the separator by one, unless the 
        # separator already only contains one point (which is the case for the 
        # `ionic_resistor` model), or if the thickness of elyte removed from 
        # the separator as less than half the thickness of a single discretized 
        # volume.  In tnis case, leave sep.npoints as is.
        if sep.n_points > 1 and self.dy_elyte >= 0.5*sep.dy:
            sep.n_points -= 1
        
        sep.dy = thickness / sep.n_points
        sep.dyInv = 1. / sep.dy
        
        return sep
    
    def elyte_potential(self, SV, j):
        
        phi_elyte = SV[self.SVptr['electrode'][self.SVptr['phi_elyte'][j]]]

        return phi_elyte

    def output(self, axs, solution, ax_offset):

        # axs[ax_offset].plot(solution[0,:]/3600, 
        #     1e6*solution[2+int(self.SVptr['thickness'])])
        # axs[ax_offset].set_ylabel(self.name+' Thickness \n($\mu$m)')

        return axs
