"""
    ionic_resistor.py

    Class file for ionic resistor separator methods. This is a very simple separator model, that considers it to have an ionic resistance, but no chemical variation.
"""

import cantera as ct
import numpy as np

def residual(SV, SVdot, an, sep, ca, params):
    """
    Define the residual for the state of the separator.

    This is a single algebraic governing equation to determine the separator electric potential.  The separator electric potential must be such that the ionic current is spatially invariant (i.e. it is constant and equal to the external applied current, for galvanostatic simulations).  

    The residual corresponding to this variable (suppose an index 'j') is of the form:
            resid[j]  = (epression equaling zero; here i_io - i_ext)
    """
    # Initialize the residual vector, assuming dSVdt = 0 (we will overwrite/
    #  replace this, as necessary)
    resid = SVdot[sep.SVptr['sep']]

    # Calculate the distance to the anode node center and the anode's electrolyte phase electric potential at the separator boundary:
    dy, phi_elyte_an = electrode_boundary_potential(SV, an, sep)

    # Calculate the electric potential that satisfies the algebraic equation:
    phi_elyte_sep = phi_elyte_an - params['i_ext']*dy/sep.sigma_io
    
    # Calculate the residual:
    resid[sep.SVptr['phi']] = (SV[sep.SVptr['sep'][sep.SVptr['phi']]] 
            - phi_elyte_sep)

    return resid

def electrode_boundary_flux(SV, ed, sep, _):
    """
    Calculate the species fluxes and ionic current between a node in the separator and one of the electrodes.
    """

    # Determine which indices are at the electrode/electrolyte boundary:
    if ed.name=='anode':
        j_ed = -1
        j_elyte = 0
    elif ed.name=='cathode':
        j_ed = 0
        j_elyte = -1

    # Initialize species fluxes:    
    N_k_elyte = np.zeros_like(ed.elyte_obj.X)

    # Elyte electric potential in electrode:
    phi_ed = SV[ed.SVptr['electrode'][ed.SVptr['phi_ed'][j_ed]]]
    phi_dl = SV[ed.SVptr['electrode'][ed.SVptr['phi_dl'][j_ed]]]
    phi_elyte_ed = phi_ed + phi_dl
    
    # Elyte electric potential in separator:
    phi_elyte_sep = SV[sep.SVptr['sep'][sep.SVptr['phi']]]
    
    # Average electronic resistance:
    dy_eff = 0.5*(sep.dy/sep.elyte_microstructure 
            + ed.dy/ed.elyte_microstructure)

    # Ionic current:
    i_io = ed.i_ext_flag*(phi_elyte_sep - phi_elyte_ed)*sep.sigma_io/dy_eff

    # Convert this to flux of the lithium ion:
    N_k_elyte[ed.index_Li] = i_io/ct.faraday/ed.elyte_obj.charges[ed.index_Li]

    return N_k_elyte, i_io

def electrode_boundary_potential(SV, ed, sep):
    """
    Calculate the effective distance between node centers at the electrode/electrolyte boundary and the electric potential in the electrolyte phase on the electrode side of this boundary.
    """
    # Elyte electric potential in anode:
    phi_ed = SV[ed.SVptr['electrode'][ed.SVptr['phi_ed']]]
    phi_dl = SV[ed.SVptr['electrode'][ed.SVptr['phi_dl']]]
    phi_elyte_ed = phi_ed + phi_dl
    
    # Effective distance between node centers, weighted by the electrolyte 
    # microstructure factors:
    dy_elyte_eff = 0.5*(sep.dy/sep.elyte_microstructure 
            + ed.dy/ed.elyte_microstructure)

    return dy_elyte_eff, phi_elyte_ed