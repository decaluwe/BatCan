"""
    single_particle_electrode.py

    Class file for dense metal (e.g. Li) electrode methods
"""

import cantera as ct
import numpy as np


def initialize(input_file, inputs, sep_inputs, counter_inputs, electrode_name, 
        params, offset):
    """
    Initialize the model.
    """    
    class electrode:
        """
        Create an electrode object representing the dense electrode
        """

        # Import relevant Cantera objects.
        bulk_obj = ct.Solution(input_file, inputs['bulk-phase'])
        elyte_obj = ct.Solution(input_file, inputs['electrolyte-phase'])
        conductor_obj = ct.Solution(input_file, inputs['conductor-phase'])
        surf_obj = ct.Interface(input_file, inputs['surf-phase'], 
            [bulk_obj, elyte_obj, conductor_obj])

        # Anode or cathode? Positive external current delivers positive charge 
        # to the anode, and removes positive charge from the cathode.
        name = electrode_name
        if name=='anode':
            i_ext_flag = -1
        elif name=='cathode':
            i_ext_flag = 1
        else:
            raise ValueError("Electrode must be an anode or a cathode.")

        # Store the species index of the Li ion in the Cantera object for the 
        # electrolyte phase:
        index_Li = elyte_obj.species_index(inputs['mobile-ion'])

        # Electrode thickness
        dy = inputs['thickness']
        # The electrode consumption rate quickly goes to zero, below a 
        # user-specified minimum thickness:
        min_thickness = inputs['minimum-thickness']

        # Interfacial surface area, per unit geometric area.
        A_surf_ratio = inputs['A_surf_ratio']

        # Inverse of the double layer capacitance, per unit interface area:
        C_dl_Inv = 1/inputs['C_dl']

        # Thickness of separator node considered as part of the anode domain.  
        # This is "subtracted" off from the total separator thickness.
        dy_elyte = inputs['dy_elyte']
        
        # Electrolyte volume fraction in the separator:
        eps_elyte = sep_inputs['eps_electrolyte']

        # Microstructure-based transport scaling factor, based on Bruggeman 
        # coefficient of -0.5:
        elyte_microstructure = eps_elyte**1.5
        
        # SV_offset specifies the index of the first SV variable for the 
        # electode (zero for anode, nVars_anode + nVars_sep for the cathode)
        SV_offset = offset
        
        # Dense Li is not capacity-limiting, in and of itself.  Rather the 
        # total amount of Li in the system is the limit. This is done in a 
        # separate routine, at a later time. Provide a large placeholder number 
        # here, so that it will not be the minimum, when evaluated later:
        capacity = 1e21
            
        # Mumber of state variables: electrode potential, electrolyte 
        # potential, thickness, electrolyte composition (n_species)
        nVars = 3 + elyte_obj.n_species

        # Load the residual function and other necessary functions, store them 
        # as methods of this class:
        from .functions import residual, voltage_lim


    # Set the Cantera object state.     
    electrode.bulk_obj.electric_potential = inputs['phi_0']
    # If the user provided an initial composition, use that, here:
    if 'X_0' in inputs:
        electrode.bulk_obj.TPX = params['T'], params['P'], inputs['X_0']
    else:
        electrode.bulk_obj.TP = params['T'], params['P']

    electrode.elyte_obj.TP = params['T'], params['P']
    electrode.surf_obj.TP = params['T'], params['P']
    electrode.conductor_obj.TP = params['T'], params['P']

    # Initialize the solution vector for the electrode domain:
    SV = np.zeros([electrode.nVars])

    # Set up pointers to specific variables in the solution vector:
    electrode.SVptr = {}
    electrode.SVptr['phi_ed'] = np.array([0])
    electrode.SVptr['phi_dl'] = np.array([1])
    electrode.SVptr['thickness'] = np.array([2])
    electrode.SVptr['C_k_elyte'] = np.arange(3, 
            3 + electrode.elyte_obj.n_species)
   
    # There is only one node, but give the pointer a shape so that SVptr
    # ['C_k_elyte'][j] accesses the pointer array:
    electrode.SVptr['C_k_elyte'].shape = (1,electrode.elyte_obj.n_species)

    # A pointer to where the SV variables for this electrode are, within the 
    # overall solution vector for the entire problem:
    electrode.SVptr['electrode'] = np.arange(offset, offset+electrode.nVars)

    # Save the SV indices of any algebraic variables:
    electrode.algvars = offset + electrode.SVptr['phi_ed'][:]
    
    # Load intial state variable values:
    SV[electrode.SVptr['phi_ed']] = inputs['phi_0']
    SV[electrode.SVptr['phi_dl']] = sep_inputs['phi_0'] - inputs['phi_0']
    SV[electrode.SVptr['thickness']] = inputs['thickness']
    SV[electrode.SVptr['C_k_elyte']] = electrode.elyte_obj.concentrations

    return SV, electrode
